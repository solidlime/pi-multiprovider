import {
  lazyStream,
  type Api,
  type ApiStreamOptions,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type AuthResult,
  type Model,
  type Provider,
  type ProviderHeaders,
  type ProviderResponse,
  type SimpleStreamOptions,
  type StreamOptions,
  type TranscriptContext,
} from '@earendil-works/pi-ai'
import { MultiProviderService } from './service.ts'
import { debugLog, shortId } from './debug.ts'
import type {
  AccountLease,
  LiftProviderOptions,
  ProviderAttemptFailure,
} from './types.ts'

type StreamKind = 'stream' | 'streamSimple'
type RequestOptions = StreamOptions & Record<string, unknown>

// Pause between same-account retries so consecutive absorbed errors give a
// briefly rate-limited or recovering backend a chance to settle.
const SAME_ACCOUNT_RETRY_DELAY_MS = 250

export interface BufferedTerminal {
  start?: AssistantMessageEvent & { type: 'start' }
  event: AssistantMessageEvent & { type: 'error' }
}

export function mergeHeaders(
  base: ProviderHeaders | undefined,
  override: ProviderHeaders | undefined,
): ProviderHeaders | undefined {
  if (base === undefined && override === undefined) return undefined
  const merged: ProviderHeaders = { ...base }
  for (const [name, value] of Object.entries(override ?? {})) {
    const lowerName = name.toLowerCase()
    for (const existingName of Object.keys(merged)) {
      if (existingName.toLowerCase() === lowerName) delete merged[existingName]
    }
    merged[name] = value
  }
  return merged
}

function applyResolvedAuth<TApi extends Api>(
  model: Model<TApi>,
  options: RequestOptions,
  resolution: AuthResult,
): { model: Model<TApi>; options: RequestOptions } {
  const nextOptions = { ...options } as RequestOptions
  delete nextOptions.apiKey
  if (resolution.auth.apiKey !== undefined) nextOptions.apiKey = resolution.auth.apiKey

  const headers = mergeHeaders(options.headers, resolution.auth.headers)
  if (headers === undefined) delete nextOptions.headers
  else nextOptions.headers = headers

  const env = resolution.env === undefined && options.env === undefined
    ? undefined
    : { ...(options.env ?? {}), ...(resolution.env ?? {}) }
  if (env === undefined) delete nextOptions.env
  else nextOptions.env = env

  return {
    model: resolution.auth.baseUrl === undefined
      ? model
      : { ...model, baseUrl: resolution.auth.baseUrl },
    options: nextOptions,
  }
}

export function failureFrom(
  error: unknown,
  response: ProviderResponse | undefined,
  outputStarted: boolean,
  assistantMessage?: ProviderAttemptFailure['assistantMessage'],
): ProviderAttemptFailure {
  const message = assistantMessage?.errorMessage
    ?? (error instanceof Error ? error.message : String(error))
  return {
    message,
    outputStarted,
    ...(response === undefined ? {} : {
      status: response.status,
      headers: response.headers,
    }),
    ...(assistantMessage === undefined ? {} : { assistantMessage }),
    ...(error === undefined ? {} : { cause: error }),
  }
}

function callProvider<TApi extends Api>(
  provider: Provider<TApi>,
  kind: StreamKind,
  model: Model<TApi>,
  context: TranscriptContext,
  options: RequestOptions,
): AssistantMessageEventStream {
  if (kind === 'streamSimple') {
    return provider.streamSimple(model, context, options as SimpleStreamOptions)
  }
  return provider.stream(model, context, options as ApiStreamOptions<TApi>)
}

export function replayTerminal(terminal: BufferedTerminal): AsyncIterable<AssistantMessageEvent> {
  return (async function* () {
    if (terminal.start !== undefined) yield terminal.start
    yield terminal.event
  })()
}

// A backend that accepts a request but never emits an event hangs the turn
// forever ("Waiting for model...") with no error to fail over on. Guard the
// first event of every attempt; once one arrives the timer is cleared, so body
// stalls stay out of scope. 30000ms is the live-tuned ceiling (the historical
// 16000ms came from the old roundrobin extension).
//
// A stall was observed to fire on merely-slow live backends (opencode-go was
// measured at ~17s to first byte), so the ceiling is tunable per deployment via
// MULTIPROVIDER_FIRST_TOKEN_TIMEOUT_MS without a release; an unset or invalid
// value keeps the 30s default.
export const FIRST_TOKEN_TIMEOUT_MS = 30_000

export function resolveFirstTokenTimeoutMs(): number {
  const raw = process.env.MULTIPROVIDER_FIRST_TOKEN_TIMEOUT_MS
  const parsed = raw === undefined ? Number.NaN : Number(raw)
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : FIRST_TOKEN_TIMEOUT_MS
}

export class FirstTokenTimeoutError extends Error {
  readonly backendId: string
  readonly timeoutMs: number

  constructor(backendId: string, timeoutMs: number) {
    super('multiprovider: backend ' + backendId + ' stalled (no first token in ' + timeoutMs + 'ms)')
    this.name = 'FirstTokenTimeoutError'
    this.backendId = backendId
    this.timeoutMs = timeoutMs
  }
}

// Wrap an attempt's event stream with the first-token watchdog. It owns the
// linkage from the caller's signal to the per-attempt controller, and on
// expiry aborts that controller (tearing the backend stream down). The timeout
// then throws, so the caller's existing pre-output failover path rotates to
// the next backend — or surfaces it once every backend is exhausted.
export async function* firstTokenWatchdog<T>(
  inner: AsyncIterable<T>,
  controller: AbortController,
  backendId: string,
  outerSignal?: AbortSignal,
  timeoutMs: number = resolveFirstTokenTimeoutMs(),
): AsyncGenerator<T> {
  const onOuterAbort = () => controller.abort(outerSignal?.reason)
  if (outerSignal?.aborted === true) controller.abort(outerSignal.reason)
  else outerSignal?.addEventListener('abort', onOuterAbort, { once: true })
  const iterator = inner[Symbol.asyncIterator]()
  let timer: ReturnType<typeof setTimeout> | undefined
  const expired = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      debugLog('watchdog.fire', { backend: backendId, timeoutMs })
      reject(new FirstTokenTimeoutError(backendId, timeoutMs))
    }, timeoutMs)
  })
  try {
    let next = await Promise.race([iterator.next(), expired])
    if (timer !== undefined) clearTimeout(timer)
    timer = undefined
    if (next.done === true) return
    yield next.value
    for (;;) {
      next = await iterator.next()
      if (next.done === true) return
      yield next.value
    }
  } catch (error) {
    if (error instanceof FirstTokenTimeoutError) controller.abort(error)
    throw error
  } finally {
    if (timer !== undefined) clearTimeout(timer)
    outerSignal?.removeEventListener('abort', onOuterAbort)
    // Hand the consumer's early exit (break / return) down to the inner
    // stream so its finally runs and the in-flight request is torn down
    // instead of outliving the released lease. Stall teardown already went
    // through controller.abort, so piling return() onto a pending next() is
    // harmless.
    try {
      void Promise.resolve(iterator.return?.()).catch(() => {})
    } catch {
      // return() threw synchronously; the stream is already being torn down,
      // so there is nothing left to unwind here.
    }
  }
}

function liftedStream<TApi extends Api, TCredentialRef>(
  provider: Provider<TApi>,
  service: MultiProviderService,
  liftOptions: LiftProviderOptions<TApi, TCredentialRef>,
  kind: StreamKind,
  model: Model<TApi>,
  context: TranscriptContext,
  options?: RequestOptions,
): AssistantMessageEventStream {
  return lazyStream(model, async () => {
    const requestOptions = { ...(options ?? {}) } as RequestOptions
    const signal = requestOptions.signal ?? new AbortController().signal
    const requestContext = { provider, model, context, requestOptions, signal }
    const attempted = new Set<string>(
      liftOptions.excludeAccountIds === undefined
        ? []
        : await liftOptions.excludeAccountIds(requestContext),
    )
    const maxAttempts = liftOptions.maxAccountAttempts ?? Number.MAX_SAFE_INTEGER
    const errorsBeforeSwitch = service.getErrorsBeforeSwitch()
    const affinityKey = liftOptions.affinityKey?.({ provider, model, context })
    let attempts = 0
    let lastRejected: BufferedTerminal | undefined
    let lastSetupError: unknown

    const attemptsStream = (async function* (): AsyncGenerator<AssistantMessageEvent> {
      debugLog('lift.stream-start', {
        pool: provider.id,
        model: model.id,
        affinityKey: shortId(affinityKey),
        excluded: [...attempted].map(shortId),
      })
      while (attempts < maxAttempts) {
        let lease: AccountLease<TCredentialRef>
        try {
          lease = await service.acquire<TCredentialRef>({
            providerId: provider.id,
            ...(affinityKey === undefined ? {} : { affinityKey }),
            excludeAccountIds: attempted,
          })
        } catch (error) {
          if (lastRejected !== undefined) {
            debugLog('lift.exhausted', { pool: provider.id, replaying: 'buffered-terminal' })
            yield* replayTerminal(lastRejected)
            return
          }
          throw lastSetupError ?? error
        }

        attempts += 1
        attempted.add(lease.accountId)
        debugLog('lift.attempt', {
          pool: provider.id,
          account: shortId(lease.accountId),
          attempt: attempts,
          affinityKey: shortId(affinityKey),
        })
        let settled = false
        let outputStarted = false
        let start: BufferedTerminal['start']
        let response: ProviderResponse | undefined
        // Per-lease outcome once the account is abandoned: 'next-account'
        // rotates to the next account inline; 'surface' ends the stream with
        // the buffered error so an external failover handler (e.g.
        // compact-then-retry) can re-enter the pool with fresh context.
        let leaseOutcome: 'next-account' | 'surface' | undefined
        let sameAccountErrors = 0

        try {
          let resolved: AuthResult
          try {
            resolved = await liftOptions.resolveAuth(lease.account, signal, requestContext)
          } catch (error) {
            lastSetupError = error
            const disposition = lease.release({
              status: 'failure',
              error: failureFrom(error, response, false),
            })
            settled = true
            if (disposition?.retryable && attempts < maxAttempts && !signal.aborted) {
              continue
            }
            throw error
          }

          const applied = applyResolvedAuth(model, requestOptions, resolved)
          const sanitized = liftOptions.sanitizeRequestOptions?.({
            provider,
            model: applied.model,
            context,
            requestOptions: applied.options,
            signal,
            account: lease.account,
            resolution: resolved,
          }) ?? applied.options
          const attemptOptions = { ...sanitized } as RequestOptions
          const onResponse = attemptOptions.onResponse
          attemptOptions.onResponse = async (nextResponse, responseModel) => {
            response = {
              status: nextResponse.status,
              headers: { ...nextResponse.headers },
            }
            await onResponse?.(nextResponse, responseModel)
          }
          if (liftOptions.disableProviderRetries !== false) attemptOptions.maxRetries = 0

          // Same-account tolerance: pre-output retryable errors are absorbed
          // on the current account until errorsBeforeSwitch is reached, so a
          // transient blip does not pay a cold-cache failover.
          while (leaseOutcome === undefined) {
            response = undefined
            start = undefined
            // Each attempt gets its own abort controller so a first-token stall
            // tears down just this stream; a caller abort propagates in.
            const attemptController = new AbortController()
            attemptOptions.signal = attemptController.signal
            const inner = callProvider(
              provider,
              kind,
              applied.model,
              context,
              attemptOptions,
            )

            let retriedSameAccount = false
            for await (const event of firstTokenWatchdog(inner, attemptController, lease.accountId, signal)) {
              if (event.type === 'start') {
                start = event
                continue
              }

              if (event.type === 'error') {
                if (event.reason === 'aborted' || signal.aborted) {
                  lease.release({ status: 'cancelled' })
                  settled = true
                } else {
                  const failure = failureFrom(
                    undefined,
                    response,
                    outputStarted,
                    event.error,
                  )
                  if (!outputStarted && sameAccountErrors + 1 < errorsBeforeSwitch) {
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
                  debugLog('lift.error-event', {
                    pool: provider.id,
                    account: shortId(lease.accountId),
                    retryable: disposition?.retryable,
                    kind: disposition?.kind,
                    outputStarted,
                    errorsOnAccount: sameAccountErrors + 1,
                  })
                  if (!outputStarted && disposition?.retryable && attempts < maxAttempts) {
                    lastRejected = {
                      ...(start === undefined ? {} : { start }),
                      event,
                    }
                    leaseOutcome = liftOptions.onFailover?.({
                      providerId: provider.id,
                      fromAccountId: lease.accountId,
                      failure,
                      errorsOnAccount: sameAccountErrors + 1,
                    }) === true
                      ? 'surface'
                      : 'next-account'
                    debugLog('lift.rotate', {
                      pool: provider.id,
                      from: shortId(lease.accountId),
                      outcome: leaseOutcome,
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
            const disposition = lease.release({
              status: 'failure',
              error: failureFrom(error, response, outputStarted),
            })
            settled = true
            if (!outputStarted && disposition?.retryable && attempts < maxAttempts) continue
            throw error
          }
        } catch (error) {
          if (!settled) {
            const disposition = signal.aborted
              ? lease.release({ status: 'cancelled' })
              : lease.release({
                  status: 'failure',
                  error: failureFrom(error, response, outputStarted),
                })
            settled = true
            debugLog('lift.throw', {
              pool: provider.id,
              account: shortId(lease.accountId),
              retryable: disposition?.retryable,
              kind: disposition?.kind,
              outputStarted,
              aborted: signal.aborted,
              message: error instanceof Error ? error.message.slice(0, 160) : String(error).slice(0, 160),
            })
            if (!outputStarted && disposition?.retryable && attempts < maxAttempts) {
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

      if (lastRejected !== undefined) {
        yield* replayTerminal(lastRejected)
        return
      }
      throw lastSetupError ?? new Error(`multiprovider: exhausted account attempts for "${provider.id}"`)
    })()

    return attemptsStream
  })
}

export function liftProvider<TApi extends Api, TCredentialRef = unknown>(
  provider: Provider<TApi>,
  service: MultiProviderService,
  options: LiftProviderOptions<TApi, TCredentialRef>,
): Provider<TApi> {
  const lifted: Provider<TApi> = {
    id: provider.id,
    name: provider.name,
    ...(provider.baseUrl === undefined ? {} : { baseUrl: provider.baseUrl }),
    ...(provider.headers === undefined ? {} : { headers: provider.headers }),
    auth: options.auth ?? provider.auth,
    getModels: () => provider.getModels(),
    ...(provider.refreshModels === undefined
      ? {}
      : { refreshModels: context => provider.refreshModels!(context) }),
    ...(provider.filterModels === undefined
      ? {}
      : { filterModels: (models, credential) => provider.filterModels!(models, credential) }),
    stream<T extends TApi>(
      model: Model<T>,
      context: TranscriptContext,
      streamOptions?: ApiStreamOptions<T>,
    ): AssistantMessageEventStream {
      return liftedStream(
        provider as Provider<T>,
        service,
        options as LiftProviderOptions<T, TCredentialRef>,
        'stream',
        model,
        context,
        streamOptions as RequestOptions | undefined,
      )
    },
    streamSimple(
      model: Model<TApi>,
      context: TranscriptContext,
      streamOptions?: SimpleStreamOptions,
    ): AssistantMessageEventStream {
      return liftedStream(
        provider,
        service,
        options,
        'streamSimple',
        model,
        context,
        streamOptions as RequestOptions | undefined,
      )
    },
  }

  return lifted
}
