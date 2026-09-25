import {
  createAssistantMessageEventStream,
  createProvider,
  normalizeContext,
  type Api,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type AuthContext,
  type Model,
  type Provider,
  type SimpleStreamOptions,
  type StopReason,
  type TranscriptContext,
} from '@earendil-works/pi-ai'
import { describe, expect, it, vi } from 'vitest'
import {
  captureVirtualModelTemplate,
  createVirtualIntegrations,
  createVirtualProvider,
  FIRST_TOKEN_TIMEOUT_MS,
  healVirtualTemplates,
  type FailoverInfo,
  liftProvider,
  MultiProviderService,
  type ProviderAccount,
  virtualSchedulerId,
  type VirtualProviderConfig,
  type VirtualProviderDependencies,
} from '../src/index.ts'

const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }

const modelA: Model<'test-api'> = {
  id: 'model-a',
  name: 'Model A',
  api: 'test-api',
  provider: 'prov-a',
  baseUrl: 'https://a.invalid',
  reasoning: false,
  thinkingLevelMap: { high: 'high-effort', off: null },
  input: ['text'],
  cost: zeroCost,
  contextWindow: 1_000,
  maxTokens: 100,
}

const modelB: Model<'test-api'> = {
  ...modelA,
  id: 'model-b',
  name: 'Model B',
  provider: 'prov-b',
  baseUrl: 'https://b.invalid',
}

const config: VirtualProviderConfig = {
  id: 'pooled',
  label: 'Pooled',
  models: [{
    id: 'ultra',
    backends: [
      { providerId: 'prov-a', modelId: 'model-a' },
      { providerId: 'prov-b', modelId: 'model-b' },
    ],
  }],
}

const context: TranscriptContext = normalizeContext({ messages: [] })

const authContext: AuthContext = {
  async env() {
    return undefined
  },
  async fileExists() {
    return false
  },
}

function message(
  stopReason: StopReason,
  options: { text?: string; errorMessage?: string } = {},
): AssistantMessage {
  return {
    role: 'assistant',
    content: options.text === undefined ? [] : [{ type: 'text', text: options.text }],
    api: 'test-api',
    provider: 'prov-a',
    model: 'model-a',
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    ...(options.errorMessage === undefined ? {} : { errorMessage: options.errorMessage }),
    timestamp: Date.now(),
  }
}

function finishWithText(stream: AssistantMessageEventStream, text: string): void {
  const done = message('stop', { text })
  stream.push({ type: 'text_start', contentIndex: 0, partial: done })
  stream.push({ type: 'text_delta', contentIndex: 0, delta: text, partial: done })
  stream.push({ type: 'text_end', contentIndex: 0, content: text, partial: done })
  stream.push({ type: 'done', reason: 'stop', message: done })
  stream.end(done)
}

function finishWithError(stream: AssistantMessageEventStream, errorMessage: string): void {
  const failed = message('error', { errorMessage })
  stream.push({ type: 'error', reason: 'error', error: failed })
  stream.end(failed)
}

function okStream(text: string): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()
  finishWithText(stream, text)
  return stream
}

function errorStream(errorMessage: string): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()
  finishWithError(stream, errorMessage)
  return stream
}

// A backend that accepts the request but never emits — the silent stall that
// used to hang the turn with no error to fail over on.
function silentStream(): AssistantMessageEventStream {
  return createAssistantMessageEventStream()
}

type Handler = (
  model: Model<'test-api'>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
) => AssistantMessageEventStream

interface Attempt {
  provider: string
  model: string
  apiKey?: string
  baseUrl?: string
}

function backend(id: string, model: Model<'test-api'>, handler: Handler): Provider<Api> {
  return createProvider<'test-api'>({
    id,
    name: id.toUpperCase(),
    auth: {
      apiKey: {
        name: 'key',
        async resolve() {
          return { auth: { apiKey: 'ambient-' + id }, source: 'test' }
        },
      },
    },
    models: [model],
    api: {
      stream: (receivedModel, _context, options) => handler(receivedModel as Model<'test-api'>, _context, options),
      streamSimple: (receivedModel, _context, options) => handler(receivedModel as Model<'test-api'>, _context, options),
    },
  })
}

function harness(
  handlers: { a: Handler; b: Handler },
  overrides: {
    missingProviders?: string[]
    affinityKey?: string
  } = {},
  options: {
    errorsBeforeSwitch?: number
    onFailover?: VirtualProviderDependencies['onFailover']
  } = {},
) {
  const attempts: Attempt[] = []
  const providers = new Map<string, Provider<Api>>([
    ['prov-a', backend('prov-a', modelA, (receivedModel, requestContext, options) => {
      const model = receivedModel as Model<'test-api'>
      attempts.push({
        provider: 'prov-a',
        model: model.id,
        ...(options?.apiKey === undefined ? {} : { apiKey: options.apiKey }),
        baseUrl: model.baseUrl,
      })
      return handlers.a(model, requestContext, options)
    })],
    ['prov-b', backend('prov-b', modelB, (receivedModel, requestContext, options) => {
      const model = receivedModel as Model<'test-api'>
      attempts.push({
        provider: 'prov-b',
        model: model.id,
        ...(options?.apiKey === undefined ? {} : { apiKey: options.apiKey }),
        baseUrl: model.baseUrl,
      })
      return handlers.b(model as Model<'test-api'>, requestContext, options)
    })],
  ])
  const service = new MultiProviderService({
    randomInt: () => 0,
    ...(options.errorsBeforeSwitch === undefined ? {} : { errorsBeforeSwitch: options.errorsBeforeSwitch }),
  })
  for (const integration of createVirtualIntegrations(config)) {
    service.registerProvider(integration)
  }
  const deps: VirtualProviderDependencies = {
    service,
    config,
    ...(options.onFailover === undefined ? {} : { onFailover: options.onFailover }),
    getAffinityKey: () => overrides.affinityKey ?? 'session-1',
    getBackingProvider: providerId =>
      overrides.missingProviders?.includes(providerId) ? undefined : providers.get(providerId),
    resolveAmbientAuth: async providerId => ({ ok: true, apiKey: 'ambient-' + providerId }),
  }
  const virtual = createVirtualProvider(deps)
  return { service, virtual, attempts, deps }
}

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = []
  for await (const event of stream) events.push(event)
  return events
}

describe('virtual providers', () => {
  it('registers one unbiased scheduler per virtual model', async () => {
    const service = new MultiProviderService()
    for (const integration of createVirtualIntegrations(config)) {
      service.registerProvider(integration)
    }
    const snapshot = await service.snapshot()
    expect(snapshot.providers).toHaveLength(1)
    expect(snapshot.providers[0]?.id).toBe(virtualSchedulerId('pooled', 'ultra'))
    expect(snapshot.providers[0]?.firstAccountBias).toBe(false)
    expect(snapshot.providers[0]?.accounts.map(account => account.id))
      .toEqual(['prov-a::model-a', 'prov-b::model-b'])
  })

  it('exposes virtual models templated from the first healthy backend', () => {
    const { virtual } = harness({ a: () => okStream('x'), b: () => okStream('x') })
    const models = virtual.getModels()
    expect(models).toHaveLength(1)
    expect(models[0]).toMatchObject({
      id: 'ultra',
      name: 'ultra',
      provider: 'pooled',
      api: 'test-api',
      baseUrl: 'https://a.invalid',
      thinkingLevelMap: { high: 'high-effort', off: null },
      contextWindow: 1_000,
    })
  })

  it('falls back to the persisted backend template before backing providers register', () => {
    const virtual = createVirtualProvider({
      service: new MultiProviderService(),
      config: {
        id: 'pooled',
        label: 'Pooled',
        models: [{
          id: 'ultra',
          backends: [{
            providerId: 'prov-a',
            modelId: 'model-a',
            template: { ...captureVirtualModelTemplate(modelA), reasoning: true },
          }],
        }],
      },
      // Mirrors extension load: no backing provider is registered yet when pi
      // snapshots enabled/resumed-session models.
      getBackingProvider: () => undefined,
      getAffinityKey: () => 'session-1',
      resolveAmbientAuth: async () => ({ ok: true }),
    })
    expect(virtual.getModels()[0]).toMatchObject({
      id: 'ultra',
      provider: 'pooled',
      api: 'test-api',
      baseUrl: 'https://a.invalid',
      reasoning: true,
      thinkingLevelMap: { high: 'high-effort', off: null },
      contextWindow: 1_000,
      maxTokens: 100,
    })
  })

  it('prefers the live backing model over the persisted template', () => {
    const virtual = createVirtualProvider({
      service: new MultiProviderService(),
      config: {
        id: 'pooled',
        label: 'Pooled',
        models: [{
          id: 'ultra',
          backends: [{
            providerId: 'prov-a',
            modelId: 'model-a',
            template: {
              ...captureVirtualModelTemplate(modelA),
              api: 'openai-completions',
              baseUrl: 'https://stored.invalid',
              reasoning: true,
              contextWindow: 5_000,
            },
          }],
        }],
      },
      getBackingProvider: providerId =>
        providerId === 'prov-a' ? backend('prov-a', modelA, () => okStream('x')) : undefined,
      getAffinityKey: () => 'session-1',
      resolveAmbientAuth: async () => ({ ok: true }),
    })
    expect(virtual.getModels()[0]).toMatchObject({
      api: 'test-api',
      baseUrl: 'https://a.invalid',
      reasoning: false,
      thinkingLevelMap: { high: 'high-effort', off: null },
      contextWindow: 1_000,
    })
  })

  it('ignores persisted templates on disabled backends', () => {
    const virtual = createVirtualProvider({
      service: new MultiProviderService(),
      config: {
        id: 'pooled',
        label: 'Pooled',
        models: [{
          id: 'ultra',
          backends: [
            { providerId: 'prov-a', modelId: 'model-a', enabled: false, template: { ...captureVirtualModelTemplate(modelA), reasoning: true } },
            { providerId: 'prov-b', modelId: 'model-b' },
          ],
        }],
      },
      getBackingProvider: () => undefined,
      getAffinityKey: () => 'session-1',
      resolveAmbientAuth: async () => ({ ok: true }),
    })
    const model = virtual.getModels()[0]!
    expect(model.reasoning).toBe(false)
    expect(model.thinkingLevelMap).toBeUndefined()
  })

  it('heals stored configs by filling missing templates from live backings', () => {
    const resolved = healVirtualTemplates(config, (providerId, modelId) =>
      providerId === 'prov-a' && modelId === 'model-a' ? captureVirtualModelTemplate(modelA) : undefined)
    expect(resolved).not.toBeUndefined()
    expect(resolved?.models[0]?.backends[0]?.template).toEqual(captureVirtualModelTemplate(modelA))
    expect(resolved?.models[0]?.backends[1]?.template).toBeUndefined()
    expect(healVirtualTemplates(config, () => undefined)).toBeUndefined()
  })

  it('round-robins backends across sessions and delegates with backing model and ambient auth', async () => {
    const { attempts, deps } = harness({
      a: () => okStream('from-a'),
      b: () => okStream('from-b'),
    })
    // One service, two sessions: the pool rotates while each session stays
    // pinned to its selected backend.
    const firstSession = createVirtualProvider({ ...deps, getAffinityKey: () => 'session-1' })
    const secondSession = createVirtualProvider({ ...deps, getAffinityKey: () => 'session-2' })
    const model = firstSession.getModels()[0]!
    const first = await collect(firstSession.stream(model, context))
    const second = await collect(secondSession.stream(model, context))
    expect(first.at(-1)).toMatchObject({ type: 'done' })
    expect(second.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
    expect(attempts.map(attempt => attempt.apiKey)).toEqual(['ambient-prov-a', 'ambient-prov-b'])
    expect(attempts.every(attempt => attempt.baseUrl?.startsWith('https://'))).toBe(true)
  })

  it('pins the session affinity key to the last healthy backend', async () => {
    const { virtual, attempts } = harness({
      a: () => okStream('from-a'),
      b: () => okStream('from-b'),
    })
    const model = virtual.getModels()[0]!
    await collect(virtual.stream(model, context))
    await collect(virtual.stream(model, context))
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-a'])
  })

  it('fails over to the next backend before output', async () => {
    const { virtual, attempts } = harness(
      { a: () => errorStream('HTTP 500 upstream'), b: () => okStream('from-b') },
      undefined,
      { errorsBeforeSwitch: 1 },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
  })

  it('rotates to the next backend on a fatal pre-stream backend error', async () => {
    // A fatal (non-retryable) internal error must not surface before the pool
    // has tried its remaining backends.
    const { virtual, attempts } = harness(
      {
        a: () => errorStream('multiprovider: provider "prov-a" has no enabled accounts'),
        b: () => okStream('from-b'),
      },
      undefined,
      { errorsBeforeSwitch: 1 },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
  })

  it('rotates to the next backend when a backend throws synchronously before output', async () => {
    const { virtual, attempts } = harness(
      {
        a: () => {
          throw new Error('multiprovider: provider "prov-a" has no enabled accounts')
        },
        b: () => okStream('from-b'),
      },
      undefined,
      { errorsBeforeSwitch: 1 },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
  })

  it('fails over when a backend provider or model is unavailable', async () => {
    const { virtual, attempts } = harness(
      { a: () => okStream('from-a'), b: () => okStream('from-b') },
      { missingProviders: ['prov-a'] },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-b'])
  })

  it('replays the terminal error once every backend is exhausted', async () => {
    const { virtual, attempts } = harness(
      { a: () => errorStream('HTTP 500'), b: () => errorStream('HTTP 503') },
      undefined,
      { errorsBeforeSwitch: 1 },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'error' })
    expect(attempts).toHaveLength(2)
  })

  it('absorbs up to three backend errors before failing over', async () => {
    const { virtual, attempts } = harness({
      a: () => errorStream('HTTP 503'),
      b: () => okStream('from-b'),
    })
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-a', 'model-a', 'model-b'])
  })

  it('surfaces the buffered error when onFailover claims the transition', async () => {
    const hookCalls: FailoverInfo[] = []
    const { virtual, attempts } = harness(
      { a: () => errorStream('HTTP 500'), b: () => errorStream('HTTP 503') },
      {},
      {
        onFailover: info => {
          hookCalls.push(info)
          return true
        },
      },
    )
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'error' })
    expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-a', 'model-a'])
    expect(hookCalls).toEqual([expect.objectContaining({
      providerId: virtualSchedulerId('pooled', 'ultra'),
      fromAccountId: 'prov-a::model-a',
      failure: expect.objectContaining({ message: 'HTTP 500', outputStarted: false }),
      errorsOnAccount: 3,
    })])
  })

  it('resolves placeholder auth deterministically even before backings register', async () => {
    // Regression: resolve used to consult pi's async availability snapshot and
    // returned undefined inside the refresh window, surfacing as
    // "No API key found for <virtual pool>" on fresh sessions/advisors.
    const { virtual } = harness(
      { a: () => okStream('x'), b: () => okStream('x') },
      { missingProviders: ['prov-a', 'prov-b'] },
    )
    const resolution = await virtual.auth.apiKey!.resolve({
      ctx: authContext,
      signal: new AbortController().signal,
    })
    expect(resolution).toMatchObject({ auth: { apiKey: 'virtual-provider' }, source: 'virtual provider' })
  })

  it('reports placeholder auth once a backend is configured', async () => {
    const { virtual } = harness({ a: () => okStream('x'), b: () => okStream('x') })
    const resolution = await virtual.auth.apiKey!.resolve({
      ctx: authContext,
      signal: new AbortController().signal,
    })
    expect(resolution).toMatchObject({ auth: { apiKey: 'virtual-provider' }, source: 'virtual provider' })
  })
})

describe('first-token watchdog', () => {
  it('fails over to the next backend when the first event never arrives', async () => {
    vi.useFakeTimers()
    try {
      const { virtual, attempts, service } = harness({
        a: () => silentStream(),
        b: () => okStream('from-b'),
      })
      const eventsPromise = collect(virtual.stream(virtual.getModels()[0]!, context))
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(FIRST_TOKEN_TIMEOUT_MS)
      const events = await eventsPromise
      expect(events.at(-1)).toMatchObject({ type: 'done' })
      expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
      const snapshot = await service.snapshot()
      expect(snapshot.providers[0]?.accounts.find(account => account.id === 'prov-a::model-a'))
        .toMatchObject({ status: 'cooldown', consecutiveFailures: 1, inFlight: 0 })
    } finally {
      vi.useRealTimers()
    }
  })

  it('aborts the stalled inner stream when the watchdog fires', async () => {
    vi.useFakeTimers()
    try {
      let stalledSignal: AbortSignal | undefined
      const { virtual, attempts } = harness({
        a: (_model, _context, options) => {
          stalledSignal = options?.signal
          return silentStream()
        },
        b: () => okStream('from-b'),
      })
      const eventsPromise = collect(virtual.stream(virtual.getModels()[0]!, context))
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(FIRST_TOKEN_TIMEOUT_MS)
      await eventsPromise
      expect(stalledSignal?.aborted).toBe(true)
      expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('surfaces a timeout error once every silent backend is exhausted', async () => {
    vi.useFakeTimers()
    try {
      const { virtual, attempts } = harness({
        a: () => silentStream(),
        b: () => silentStream(),
      })
      const eventsPromise = collect(virtual.stream(virtual.getModels()[0]!, context))
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(FIRST_TOKEN_TIMEOUT_MS)
      await vi.advanceTimersByTimeAsync(FIRST_TOKEN_TIMEOUT_MS)
      const events = await eventsPromise
      expect(attempts.map(attempt => attempt.model)).toEqual(['model-a', 'model-b'])
      const last = events.at(-1)
      expect(last).toMatchObject({ type: 'error' })
      expect((last as { error: { errorMessage: string } }).error.errorMessage)
        .toContain('stalled (no first token in ' + FIRST_TOKEN_TIMEOUT_MS + 'ms)')
    } finally {
      vi.useRealTimers()
    }
  })

  it('clears the watchdog once the first event arrives', async () => {
    vi.useFakeTimers()
    try {
      const { virtual } = harness({ a: () => okStream('from-a'), b: () => okStream('from-b') })
      const eventsPromise = collect(virtual.stream(virtual.getModels()[0]!, context))
      await vi.advanceTimersByTimeAsync(0)
      const events = await eventsPromise
      expect(events.at(-1)).toMatchObject({ type: 'done' })
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })
})

// The real-device shape: a virtual pool over a backing provider that is itself
// pooled (registered on the same service, with its own watchdog + rotation).
// The inner pool must own stall recovery; a second watchdog on the virtual
// side would fire first and cancel the inner rotation.
describe('nested watchdog (virtual over a pooled backing provider)', () => {
  const pooledModel: Model<'test-api'> = {
    ...modelA,
    id: 'same-model',
    name: 'Same Model',
    provider: 'pooled-backend',
    baseUrl: 'https://pooled.invalid',
  }

  const backendAccounts: ProviderAccount<string>[] = [
    { id: 'acct-slow', label: 'Slow', authKind: 'api-key', credentialRef: 'acct-slow' },
    { id: 'acct-fast', label: 'Fast', authKind: 'api-key', credentialRef: 'acct-fast' },
  ]

  function nested(
    handler: (accountId: string, options?: SimpleStreamOptions) => AssistantMessageEventStream,
  ) {
    const service = new MultiProviderService({ randomInt: () => 0 })
    service.registerProvider({
      id: 'pooled-backend',
      label: 'Pooled Backend',
      selectionBias: 'first-account',
      accounts: () => backendAccounts,
    })
    const base = createProvider<'test-api'>({
      id: 'pooled-backend',
      name: 'Pooled Backend',
      auth: {
        apiKey: {
          name: 'key',
          async resolve() {
            return { auth: { apiKey: 'outer-placeholder' }, source: 'test' }
          },
        },
      },
      models: [pooledModel],
      api: {
        stream: (_model, _context, options) =>
          handler(String(options?.apiKey).replace(/^key-/, ''), options),
        streamSimple: (_model, _context, options) =>
          handler(String(options?.apiKey).replace(/^key-/, ''), options),
      },
    })
    const lifted = liftProvider<'test-api', string>(base, service, {
      resolveAuth: account => ({ auth: { apiKey: 'key-' + account.credentialRef } }),
    })
    const config: VirtualProviderConfig = {
      id: 'outer',
      label: 'Outer',
      models: [{ id: 'ultra', backends: [{ providerId: 'pooled-backend', modelId: pooledModel.id }] }],
    }
    for (const integration of createVirtualIntegrations(config)) service.registerProvider(integration)
    const virtual = createVirtualProvider({
      service,
      config,
      getAffinityKey: () => 'session-1',
      getBackingProvider: providerId => (providerId === 'pooled-backend' ? lifted : undefined),
      resolveAmbientAuth: async () => ({ ok: true, apiKey: 'ambient' }),
    })
    return { service, virtual }
  }

  it('lets the pooled backend rotate on a stall instead of surfacing it', async () => {
    vi.useFakeTimers()
    try {
      const accountsCalled: string[] = []
      const { virtual } = nested(accountId => {
        accountsCalled.push(accountId)
        return accountId === 'acct-slow' ? silentStream() : okStream('from-acct-fast')
      })
      const eventsPromise = collect(virtual.stream(virtual.getModels()[0]!, context))
      await vi.advanceTimersByTimeAsync(0)
      await vi.advanceTimersByTimeAsync(FIRST_TOKEN_TIMEOUT_MS)
      const events = await eventsPromise
      expect(events.at(-1)).toMatchObject({ type: 'done' })
      expect(accountsCalled).toEqual(['acct-slow', 'acct-fast'])
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not re-enter a pooled backend whose accounts are already exhausted', async () => {
    // The pooled backend below owns one account that always fails, so every
    // inner attempt already burns errorsBeforeSwitch (default 3) tries before
    // the backend surfaces an error. The outer virtual layer must rotate on
    // that first error instead of replaying the whole subtree three times.
    const service = new MultiProviderService({ randomInt: () => 0 })
    let pooledCalls = 0
    const pooledOnlyModel: Model<'test-api'> = { ...pooledModel, id: 'only-model' }
    service.registerProvider({
      id: 'pooled-backend',
      label: 'Pooled Backend',
      accounts: () => [{ id: 'only', label: 'Only', authKind: 'api-key', credentialRef: 'only' }],
      // Zero cooldown keeps the single account immediately re-available, so an
      // outer retry really does re-run the whole subtree (as it does in
      // production while a backing account's short stall cooldown expires).
      classifyFailure: () => ({ kind: 'transient', retryable: true, cooldownMs: 0 }),
    })
    const pooledBase = createProvider<'test-api'>({
      id: 'pooled-backend',
      name: 'Pooled Backend',
      auth: {
        apiKey: {
          name: 'key',
          async resolve() {
            return { auth: { apiKey: 'placeholder' }, source: 'test' }
          },
        },
      },
      models: [pooledOnlyModel],
      api: {
        stream: () => { pooledCalls += 1; return errorStream('HTTP 503 upstream') },
        streamSimple: () => { pooledCalls += 1; return errorStream('HTTP 503 upstream') },
      },
    })
    const lifted = liftProvider<'test-api', string>(pooledBase, service, {
      resolveAuth: account => ({ auth: { apiKey: 'key-' + account.credentialRef } }),
    })
    const liveModel: Model<'test-api'> = { ...modelB, id: 'live-model', provider: 'prov-live' }
    let liveCalls = 0
    const liveProvider = backend('prov-live', liveModel, () => {
      liveCalls += 1
      return okStream('from-live')
    })
    const nestedConfig: VirtualProviderConfig = {
      id: 'outer-nested',
      label: 'Outer',
      models: [{
        id: 'ultra',
        backends: [
          { providerId: 'pooled-backend', modelId: pooledOnlyModel.id },
          { providerId: 'prov-live', modelId: liveModel.id },
        ],
      }],
    }
    for (const integration of createVirtualIntegrations(nestedConfig)) service.registerProvider(integration)
    const virtual = createVirtualProvider({
      service,
      config: nestedConfig,
      getAffinityKey: () => 'session-1',
      getBackingProvider: providerId => (providerId === 'pooled-backend' ? lifted : liveProvider),
      resolveAmbientAuth: async () => ({ ok: false, error: 'test: no ambient auth' }),
    })
    const events = await collect(virtual.stream(virtual.getModels()[0]!, context))
    expect(events.at(-1)).toMatchObject({ type: 'done' })
    expect(pooledCalls).toBe(3)
    expect(liveCalls).toBe(1)
  })

  it('propagates a caller cancel into the pooled backend', async () => {
    let innerSignal: AbortSignal | undefined
    let signalCalled: () => void = () => {}
    const called = new Promise<void>(resolve => { signalCalled = resolve })
    const { virtual } = nested((_accountId, options) => {
      innerSignal = options?.signal
      signalCalled()
      const stream = createAssistantMessageEventStream()
      const fail = () => {
        const failed = message('error', { errorMessage: 'cancelled' })
        stream.push({ type: 'error', reason: 'aborted', error: failed })
        stream.end(failed)
      }
      if (options?.signal?.aborted === true) fail()
      else options?.signal?.addEventListener('abort', fail, { once: true })
      return stream
    })
    const controller = new AbortController()
    const eventsPromise = collect(
      virtual.stream(virtual.getModels()[0]!, context, { signal: controller.signal }),
    )
    await called
    expect(innerSignal?.aborted).toBe(false)
    controller.abort()
    const events = await eventsPromise
    expect(innerSignal?.aborted).toBe(true)
    expect(events.at(-1)).toMatchObject({ type: 'error' })
  })
})
