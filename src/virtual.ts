import {
  lazyStream,
  type Api,
  type ApiStreamOptions,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Model,
  type Provider,
  type ProviderHeaders,
  type ProviderResponse,
  type SimpleStreamOptions,
  type StreamOptions,
  type TranscriptContext,
} from '@earendil-works/pi-ai'
import {
  failureFrom,
  firstTokenWatchdog,
  mergeHeaders,
  replayTerminal,
  type BufferedTerminal,
} from './lift.ts'
import { debugLog, shortId } from './debug.ts'
import { defaultDisposition } from './service.ts'
import type { MultiProviderService } from './service.ts'
import type {
  AccountLease,
  FailoverInfo,
  ProviderAttemptFailure,
  ProviderRegistration,
  SelectionBias,
  VirtualBackend,
  VirtualModelConfig,
  VirtualModelTemplate,
  VirtualProviderConfig,
} from './types.ts'

type StreamKind = 'stream' | 'streamSimple'
type RequestOptions = StreamOptions & Record<string, unknown>

// Pause between same-account retries so consecutive absorbed errors give a
// briefly rate-limited or recovering backend a chance to settle.
const SAME_ACCOUNT_RETRY_DELAY_MS = 250

// Virtual provider ids, virtual model ids, and provider/model ids must not
// contain this separator: it composes scheduler ids and backend account ids.
export const VIRTUAL_ID_SEPARATOR = '::'
export const BACKEND_UNAVAILABLE_PREFIX = 'multiprovider: virtual backend unavailable'

// Placeholder credential the virtual provider reports to the host so its
// models pass auth-availability checks; real auth resolves per attempt at the
// backing provider layer.
const VIRTUAL_PLACEHOLDER_API_KEY = 'virtual-provider'

export function virtualSchedulerId(virtualProviderId: string, modelId: string): string {
  return virtualProviderId + VIRTUAL_ID_SEPARATOR + modelId
}

export function virtualBackendAccountId(
  backend: Pick<VirtualBackend, 'providerId' | 'modelId'>,
): string {
  return backend.providerId + VIRTUAL_ID_SEPARATOR + backend.modelId
}

// Ambient (host-registered) auth for a backing provider, resolved per attempt.
// A failed resolution does not abort the attempt: backing providers with their
// own multiprovider pool resolve stored credentials themselves.
export type AmbientAuthResolution =
  | {
      ok: true
      apiKey?: string
      headers?: ProviderHeaders
      baseUrl?: string
      env?: Record<string, string>
    }
  | { ok: false; error: string }

export interface VirtualProviderDependencies {
  service: MultiProviderService
  config: VirtualProviderConfig
  getAffinityKey: () => string
  getBackingProvider: (providerId: string) => Provider<Api> | undefined
  resolveAmbientAuth: (
    providerId: string,
    model: Model<Api>,
    signal: AbortSignal,
  ) => Promise<AmbientAuthResolution>
  maxAccountAttempts?: number
  onFailover?: (info: FailoverInfo) => boolean | void
}

export interface VirtualIntegrationOptions {
  getProviderLabel?: (providerId: string) => string | undefined
  maxAccountAttempts?: number
  // A backing provider registered on the same service is itself a pool with
  // its own per-account cooldowns. Stacking the virtual scheduler's kind-based
  // cooldown on the backend as a whole seals it off long after every inner
  // account recovers, so pooled backends get an explicit 0ms cooldown and stay
  // immediately re-selectable. Plain backends keep their own cooldown.
  isPooledBackend?: (providerId: string) => boolean
}

// One scheduler registration per virtual model: each model's backends pool
// independently, and every virtual pool rotates with selectionBias 'none' so
// sessions spread evenly across providers (no first-provider favoritism).
export function createVirtualIntegrations(
  config: VirtualProviderConfig,
  options: VirtualIntegrationOptions = {},
): ProviderRegistration<VirtualBackend>[] {
  return config.models.map(model => ({
    id: virtualSchedulerId(config.id, model.id),
    label: config.label + ' · ' + (model.label ?? model.id),
    selectionBias: 'none' as SelectionBias,
    accounts: () =>
      model.backends
        .filter(backend => backend.enabled !== false)
        .map(backend => ({
          id: virtualBackendAccountId(backend),
          label: (options.getProviderLabel?.(backend.providerId) ?? backend.providerId) + ' · ' + backend.modelId,
          authKind: 'custom' as const,
          credentialRef: backend,
          weight: backend.weight ?? 1,
          metadata: { virtual: true, providerId: backend.providerId, modelId: backend.modelId },
        })),
    classifyFailure: (failure: ProviderAttemptFailure, account) => {
      const message = failure.message.toLowerCase()
      const transient = failure.message.startsWith(BACKEND_UNAVAILABLE_PREFIX)
        || /fetch failed|network|econn(?:aborted|refused|reset)|enotfound|etimedout|socket hang up|not configured/.test(message)
      const classification = transient
        ? { kind: 'transient' as const, retryable: true }
        : undefined
      if (options.isPooledBackend?.(account.credentialRef.providerId) === true) {
        return {
          ...(classification ?? defaultDisposition(failure)),
          cooldownMs: 0,
        }
      }
      return classification
    },
    ...(options.maxAccountAttempts === undefined ? {} : { maxAccountAttempts: options.maxAccountAttempts }),
  }))
}

function resolveTarget(
  dependencies: VirtualProviderDependencies,
  backend: VirtualBackend,
): { provider: Provider<Api>; model: Model<Api> } | string {
  const provider = dependencies.getBackingProvider(backend.providerId)
  if (provider === undefined) {
    return BACKEND_UNAVAILABLE_PREFIX + ': provider "' + backend.providerId + '" is not registered'
  }
  const model = provider.getModels().find(candidate => candidate.id === backend.modelId)
  if (model === undefined) {
    return BACKEND_UNAVAILABLE_PREFIX + ': provider "' + backend.providerId + '" has no model "' + backend.modelId + '"'
  }
  return { provider, model }
}

// In-process subagents can fire their first request before the host finishes
// installing backing providers (session_start race). Wait briefly for the
// provider to appear instead of failing the attempt instantly.
async function resolveTargetWait(
  dependencies: VirtualProviderDependencies,
  backend: VirtualBackend,
  signal: AbortSignal,
): Promise<{ provider: Provider<Api>; model: Model<Api> } | string> {
  let target = resolveTarget(dependencies, backend)
  if (typeof target !== 'string' || !target.endsWith('is not registered')) return target
  const deadline = Date.now() + 2500
  while (Date.now() < deadline && !signal.aborted) {
    await new Promise(resolve => setTimeout(resolve, 200))
    target = resolveTarget(dependencies, backend)
    if (typeof target !== 'string' || !target.endsWith('is not registered')) return target
  }
  return target
}

function virtualStream<TApi extends Api>(
  dependencies: VirtualProviderDependencies,
  kind: StreamKind,
  model: Model<TApi>,
  context: TranscriptContext,
  options?: RequestOptions,
): AssistantMessageEventStream {
  const { config, service } = dependencies
  return lazyStream(model, async () => {
    const requestOptions = { ...(options ?? {}) } as RequestOptions
    const signal = requestOptions.signal ?? new AbortController().signal
    const schedulerId = virtualSchedulerId(config.id, model.id)
    const affinityKey = dependencies.getAffinityKey()
    const attempted = new Set<string>()
    const maxAttempts = dependencies.maxAccountAttempts ?? Number.MAX_SAFE_INTEGER
    const errorsBeforeSwitch = service.getErrorsBeforeSwitch()
    let attempts = 0
    let lastTerminal: BufferedTerminal | undefined
    let lastSetupError: unknown

    const attemptsStream = (async function* (): AsyncGenerator<AssistantMessageEvent> {
      debugLog('virtual.stream-start', {
        virtual: config.id,
        model: model.id,
        affinityKey: shortId(affinityKey),
      })
      while (attempts < maxAttempts) {
        let lease: AccountLease<VirtualBackend>
        try {
          lease = await service.acquire<VirtualBackend>({
            providerId: schedulerId,
            affinityKey,
            excludeAccountIds: attempted,
          })
        } catch (error) {
          if (lastTerminal !== undefined) {
            yield* replayTerminal(lastTerminal)
            return
          }
          throw lastSetupError ?? error
        }

        attempts += 1
        attempted.add(lease.accountId)
        const backend = lease.credentialRef
        debugLog('virtual.attempt', {
          virtual: config.id,
          model: model.id,
          backend: lease.accountId,
          attempt: attempts,
          affinityKey: shortId(affinityKey),
        })
        let settled = false
        let outputStarted = false
        let start: BufferedTerminal['start']
        let response: ProviderResponse | undefined
        // Per-lease outcome once the backend is abandoned: 'next-account'
        // rotates to the next backend inline; 'surface' ends the stream with
        // the buffered error so an external failover handler (e.g.
        // compact-then-retry) can re-enter the pool with fresh context.
        let leaseOutcome: 'next-account' | 'surface' | undefined
        let sameAccountErrors = 0

        try {
          const target = await resolveTargetWait(dependencies, backend, signal)
          if (typeof target === 'string') {
            lastSetupError = new Error(target)
            debugLog('virtual.backend-unavailable', { virtual: config.id, backend: lease.accountId, detail: target.slice(0, 160) })
            lease.release({
              status: 'failure',
              error: { message: target, outputStarted: false },
            })
            settled = true
            // A pre-output setup failure rotates to the next untried backend
            // regardless of disposition; the acquire above throws once none
            // remain, surfacing lastSetupError.
            if (attempts < maxAttempts && !signal.aborted) continue
            throw lastSetupError
          }

          const ambient = await dependencies.resolveAmbientAuth(backend.providerId, target.model, signal)
          const attemptOptions = { ...requestOptions } as RequestOptions
          // The host resolves auth for the virtual provider itself (a
          // placeholder key); backing auth comes from the ambient layer below
          // or from the backing provider's own integration.
          delete attemptOptions.apiKey
          if (ambient.ok) {
            if (ambient.apiKey !== undefined) attemptOptions.apiKey = ambient.apiKey
            const headers = mergeHeaders(requestOptions.headers, ambient.headers)
            if (headers !== undefined) attemptOptions.headers = headers
            const env = ambient.env === undefined && requestOptions.env === undefined
              ? undefined
              : { ...(requestOptions.env ?? {}), ...(ambient.env ?? {}) }
            if (env !== undefined) attemptOptions.env = env
          }
          const streamModel: Model<Api> = ambient.ok && ambient.baseUrl !== undefined
            ? { ...target.model, baseUrl: ambient.baseUrl }
            : target.model
          const onResponse = attemptOptions.onResponse
          attemptOptions.onResponse = async (nextResponse, responseModel) => {
            response = {
              status: nextResponse.status,
              headers: { ...nextResponse.headers },
            }
            await onResponse?.(nextResponse, responseModel)
          }
          attemptOptions.maxRetries = 0

          // Same-account tolerance: pre-output retryable errors are absorbed
          // on the current backend until errorsBeforeSwitch is reached, so a
          // transient blip does not pay a cold-cache failover.
          while (leaseOutcome === undefined) {
            response = undefined
            start = undefined
            // Each attempt gets its own abort controller so a first-token stall
            // tears down just this stream; a caller abort propagates in. A
            // pooled backing provider (registered on this same service) owns
            // its own watchdog and rotation, so wrapping it in a second one
            // lets this outer timer fire first and cancel the inner rotation;
            // hand it the caller signal untouched instead.
            const attemptController = new AbortController()
            const pooledBackend = service.hasProvider(backend.providerId)
            debugLog('virtual.stream-call', {
              virtual: config.id,
              backend: lease.accountId,
              pooled: pooledBackend,
              watchDog: pooledBackend ? 'inner' : 'outer',
            })
            attemptOptions.signal = pooledBackend ? signal : attemptController.signal
            const inner = kind === 'streamSimple'
              ? target.provider.streamSimple(streamModel, context, attemptOptions as SimpleStreamOptions)
              : target.provider.stream(streamModel, context, attemptOptions as ApiStreamOptions<Api>)
            const guarded = pooledBackend
              ? inner
              : firstTokenWatchdog(inner, attemptController, lease.accountId, signal)

            let retriedSameAccount = false
            for await (const event of guarded) {
              if (event.type === 'start') {
                start = event
                continue
              }

              if (event.type === 'error') {
                if (event.reason === 'aborted' || signal.aborted) {
                  lease.release({ status: 'cancelled' })
                  settled = true
                } else {
                  const failure = failureFrom(undefined, response, outputStarted, event.error)
                  // A pooled backend owns its own per-account tolerance and
                  // rotation, so an error it surfaces already means every
                  // account underneath was tried. Re-entering the whole subtree
                  // here just replays the same stalls (and multiplies the
                  // cooldown each account accrues), so only a plain backend
                  // gets the outer same-account tolerance.
                  if (!outputStarted && !pooledBackend && sameAccountErrors + 1 < errorsBeforeSwitch) {
                    sameAccountErrors += 1
                    await new Promise(resolve => { setTimeout(resolve, SAME_ACCOUNT_RETRY_DELAY_MS) })
                    if (signal.aborted) {
                      lease.release({ status: 'cancelled' })
                      settled = true
                      return
                    }
                    retriedSameAccount = true
                    break
                  }
                  const disposition = lease.release({ status: 'failure', error: failure })
                  settled = true
                  debugLog('virtual.error-event', {
                    virtual: config.id,
                    backend: lease.accountId,
                    retryable: disposition?.retryable,
                    kind: disposition?.kind,
                    outputStarted,
                    errorsOnAccount: sameAccountErrors + 1,
                  })
                  if (!outputStarted && attempts < maxAttempts) {
                    lastTerminal = {
                      ...(start === undefined ? {} : { start }),
                      event,
                    }
                    // Every pre-output failure rotates to the next untried
                    // backend; a retryable one may instead be claimed by an
                    // external failover handler (compact-then-retry), while a
                    // fatal internal error (missing accounts, exhausted
                    // backends) must not be surfaced before rotating.
                    leaseOutcome = disposition?.retryable
                      && dependencies.onFailover?.({
                        providerId: schedulerId,
                        fromAccountId: lease.accountId,
                        failure,
                        errorsOnAccount: sameAccountErrors + 1,
                      }) === true
                      ? 'surface'
                      : 'next-account'
                    debugLog('virtual.rotate', {
                      virtual: config.id,
                      from: lease.accountId,
                      outcome: leaseOutcome,
                      message: event.error?.errorMessage?.slice(0, 160),
                    })
                    break
                  }
                }

                if (!outputStarted && start !== undefined) yield start
                yield event
                return
              }

              if (event.type === 'done') {
                lease.release({ status: 'success' })
                settled = true
                if (!outputStarted && start !== undefined) yield start
                yield event
                return
              }

              if (!outputStarted) {
                outputStarted = true
                if (start !== undefined) yield start
              }
              yield event
            }

            if (!retriedSameAccount) break
          }

          if (leaseOutcome === 'next-account') continue

          if (!settled) {
            const error = new Error('Provider stream ended without a terminal event')
            lastSetupError = error
            lease.release({
              status: 'failure',
              error: failureFrom(error, response, outputStarted),
            })
            settled = true
            if (!outputStarted && attempts < maxAttempts && !signal.aborted) continue
            throw error
          }
        } catch (error) {
          if (!settled) {
            if (signal.aborted) lease.release({ status: 'cancelled' })
            else lease.release({
              status: 'failure',
              error: failureFrom(error, response, outputStarted),
            })
            settled = true
            // A synchronous backend throw before output rotates too, except
            // when the caller aborted (cancel) — that surfaces immediately.
            debugLog('virtual.throw', {
              virtual: config.id,
              backend: lease.accountId,
              outputStarted,
              aborted: signal.aborted,
              message: error instanceof Error ? error.message.slice(0, 160) : String(error).slice(0, 160),
            })
            if (!outputStarted && attempts < maxAttempts && !signal.aborted) {
              lastSetupError = error
              continue
            }
          }
          throw error
        } finally {
          if (!settled) lease.release({ status: 'cancelled' })
        }

        if (leaseOutcome === 'surface') break
      }

      if (lastTerminal !== undefined) {
        debugLog('virtual.exhausted', { virtual: config.id, replaying: 'buffered-terminal' })
        yield* replayTerminal(lastTerminal)
        return
      }
      throw lastSetupError ?? new Error('multiprovider: exhausted virtual backends for "' + config.id + '"')
    })()

    return attemptsStream
  })
}

// Snapshot the host-facing metadata a virtual model must advertise so pi can
// clamp thinking levels and size context before the backing provider
// registers (pi snapshots enabled/resumed-session models right after
// extension load).
export function captureVirtualModelTemplate(model: Model<Api>): VirtualModelTemplate {
  return {
    api: model.api,
    baseUrl: model.baseUrl,
    reasoning: model.reasoning,
    ...(model.thinkingLevelMap === undefined ? {} : { thinkingLevelMap: model.thinkingLevelMap }),
    input: [...model.input],
    cost: { ...model.cost },
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
  }
}

// Fill in missing persisted templates from live backing models. Returns a
// cloned config when anything was added, else undefined — callers persist the
// healed config so the next extension load snapshots virtual models with
// correct thinking metadata without waiting for an editor save.
export function healVirtualTemplates(
  config: VirtualProviderConfig,
  resolveTemplate: (providerId: string, modelId: string) => VirtualModelTemplate | undefined,
): VirtualProviderConfig | undefined {
  let added = false
  const models = config.models.map(model => ({
    ...model,
    backends: model.backends.map(backend => {
      if (backend.enabled === false || backend.template !== undefined) return backend
      const template = resolveTemplate(backend.providerId, backend.modelId)
      if (template === undefined) return backend
      added = true
      return { ...backend, template }
    }),
  }))
  return added ? { ...config, models } : undefined
}

export function createVirtualProvider(dependencies: VirtualProviderDependencies): Provider<Api> {
  const { config } = dependencies

  const virtualModel = (model: VirtualModelConfig): Model<Api> => {
    let template: Model<Api> | undefined
    for (const backend of model.backends) {
      if (backend.enabled === false) continue
      const candidate = dependencies
        .getBackingProvider(backend.providerId)
        ?.getModels()
        .find(item => item.id === backend.modelId)
      if (candidate !== undefined) {
        template = candidate
        break
      }
    }
    // Live backings win. Before they register (extension load, when pi already
    // snapshots enabled/resumed-session models), fall back to the template
    // captured at backend-pick time so thinking support and context metadata
    // do not depend on provider registration order.
    const source: Model<Api> | VirtualModelTemplate | undefined = template
      ?? model.backends.find(backend => backend.enabled !== false && backend.template !== undefined)
        ?.template
    return {
      id: model.id,
      name: model.label ?? model.id,
      api: source?.api ?? 'openai-completions',
      provider: config.id,
      baseUrl: source?.baseUrl ?? '',
      reasoning: source?.reasoning ?? false,
      ...(source?.thinkingLevelMap === undefined
        ? {}
        : { thinkingLevelMap: source.thinkingLevelMap }),
      input: source?.input ?? ['text'],
      cost: source?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: source?.contextWindow ?? 128_000,
      maxTokens: source?.maxTokens ?? 8_192,
    }
  }

  const provider: Provider<Api> = {
    id: config.id,
    name: config.label,
    auth: {
      apiKey: {
        name: config.label + ' (virtual)',
        async resolve() {
          const backends = config.models
            .flatMap(model => model.backends)
            .filter(backend => backend.enabled !== false)
          if (backends.length === 0) return undefined
          // ponytail: key resolution must be deterministic — a placeholder is
          // issued whenever an enabled backend exists, and real credential
          // checks happen at stream time via resolveAmbientAuth. Basing this
          // on pi's async availability snapshot made resolve() fail inside
          // the refresh window ("No API key found for <virtual pool>").
          return { auth: { apiKey: VIRTUAL_PLACEHOLDER_API_KEY }, source: 'virtual provider' }
        },
      },
    },
    getModels: () => config.models.map(virtualModel),
    stream<T extends Api>(
      model: Model<T>,
      context: TranscriptContext,
      streamOptions?: ApiStreamOptions<T>,
    ): AssistantMessageEventStream {
      return virtualStream(dependencies, 'stream', model, context, streamOptions as RequestOptions | undefined)
    },
    streamSimple(
      model: Model<Api>,
      context: TranscriptContext,
      streamOptions?: SimpleStreamOptions,
    ): AssistantMessageEventStream {
      return virtualStream(dependencies, 'streamSimple', model, context, streamOptions as RequestOptions | undefined)
    },
  }
  return provider
}
