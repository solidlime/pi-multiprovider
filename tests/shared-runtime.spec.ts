import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createProvider, type Api, type Model, type Provider } from '@earendil-works/pi-ai'
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent'
import type { VirtualProviderConfig } from '../src/index.ts'
import { beforeEach, describe, expect, it } from 'vitest'

// pi-web shares one ModelRuntime/modelRegistry across every session, so a
// second session's shutdown must not yank a virtual provider the first session
// is still streaming through. The harness models exactly that: a single
// registry map that every session context reads and writes.
const agentDir = mkdtempSync(join(tmpdir(), 'pi-multiprovider-shared-'))
process.env.PI_CODING_AGENT_DIR = agentDir

const { MultiAuthStore } = await import('../src/index.ts')
const { default: multiprovider } = await import('../extensions/multiprovider.ts')

const model: Model<'probe-api'> = {
  id: 'probe-model',
  name: 'Probe Model',
  api: 'probe-api',
  provider: 'example',
  baseUrl: 'https://example.invalid',
  reasoning: false,
  input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 1_000,
  maxTokens: 100,
}

const base = createProvider<'probe-api'>({
  id: 'example',
  name: 'Example',
  auth: { apiKey: { name: 'Example API key', async resolve() { return undefined } } },
  models: [model],
  api: {
    stream() { throw new Error('not used') },
    streamSimple() { throw new Error('not used') },
  },
}) as Provider<'probe-api'>

const store = new MultiAuthStore()

function virtualConfig(id: string): VirtualProviderConfig {
  return {
    id,
    label: id,
    models: [{ id: 'ultra', backends: [{ providerId: base.id, modelId: model.id }] }],
  }
}

interface ExtensionHarness {
  /** The shared runtime registry, as ModelRuntime.modelRegistry would expose it. */
  registry: Map<string, Provider<Api>>
  registerCalls: string[]
  unregisterCalls: string[]
  start(sessionId?: string): Promise<void>
  beforeAgentStart(sessionId?: string): Promise<void>
  shutdown(sessionId?: string): Promise<void>
}

// Boots the real bundled extension against duck-typed Pi APIs over one shared
// registry; session contexts only differ by session id.
async function launch(): Promise<ExtensionHarness> {
  const entries: unknown[] = []
  const registry = new Map<string, Provider<Api>>()
  const registerCalls: string[] = []
  const unregisterCalls: string[] = []
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => Promise<void> | void)[]>()
  const bus = new Map<string, Set<(value: unknown) => void>>()

  const contextFor = (sessionId: string): ExtensionContext => ({
    ui: {
      notify() {},
      async select() { return undefined },
      async input() { return undefined },
      async confirm() { return false },
      async custom() { return undefined },
    },
    mode: 'tui',
    hasUI: true,
    cwd: process.cwd(),
    sessionManager: { getSessionId: () => sessionId, getEntries: () => entries },
    modelRegistry: {
      getProvider: (id: string) => registry.get(id) ?? (id === base.id ? base : undefined),
      getAll: () => [model],
      getApiKeyAndHeaders: async () => ({ ok: true }),
    },
    model: undefined as Model<'probe-api'> | undefined,
    isIdle: () => true,
    isProjectTrusted: () => true,
    signal: undefined,
    abort() {},
    hasPendingMessages: () => false,
    shutdown() {},
    getContextUsage: () => undefined,
    compact() {},
    getSystemPrompt: () => '',
  }) as unknown as ExtensionContext

  const pi = {
    events: {
      emit(name: string, value: unknown) { for (const callback of bus.get(name) ?? []) callback(value) },
      on(name: string, callback: (value: unknown) => void) {
        const listeners = bus.get(name) ?? new Set<(value: unknown) => void>()
        listeners.add(callback)
        bus.set(name, listeners)
        return () => listeners.delete(callback)
      },
    },
    on(name: string, handler: (event: unknown, ctx: unknown) => Promise<void> | void) {
      const list = handlers.get(name) ?? []
      list.push(handler)
      handlers.set(name, list)
    },
    registerProvider(provider: Provider<Api>) {
      registerCalls.push(provider.id)
      registry.set(provider.id, provider)
    },
    unregisterProvider(providerId: string) {
      unregisterCalls.push(providerId)
      registry.delete(providerId)
    },
    getAllTools: () => [],
    registerCommand() {},
    appendEntry(customType: string, data?: unknown) {
      entries.push({
        type: 'custom',
        customType,
        data,
        id: 'entry-' + (entries.length + 1),
        parentId: null,
        timestamp: new Date().toISOString(),
      })
    },
  }

  const fire = async (name: string, ctx: ExtensionContext): Promise<void> => {
    for (const handler of handlers.get(name) ?? []) {
      await handler({ type: name, reason: 'startup' }, ctx)
    }
  }

  await multiprovider(pi as unknown as ExtensionAPI)

  return {
    registry,
    registerCalls,
    unregisterCalls,
    start: sessionId => fire('session_start', contextFor(sessionId ?? 'parent')),
    beforeAgentStart: sessionId => fire('before_agent_start', contextFor(sessionId ?? 'parent')),
    shutdown: sessionId => fire('session_shutdown', contextFor(sessionId ?? 'child')),
  }
}

beforeEach(async () => {
  for (const config of await store.listVirtualProviders()) {
    await store.removeVirtualProvider(config.id)
  }
})

describe('shared runtime virtual provider lifetime', () => {
  it('keeps a foreign provider object installed when this instance shuts down', async () => {
    const runtime = await launch()
    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.start()
    const ours = runtime.registry.get('pooled')
    expect(ours).toBeDefined()

    // Another session installed its own object under the same id after us.
    const foreign = createProvider<'probe-api'>({
      id: 'pooled',
      name: 'foreign',
      auth: { apiKey: { name: 'k', async resolve() { return undefined } } },
      models: [model],
      api: { stream() { throw new Error('not used') }, streamSimple() { throw new Error('not used') } },
    }) as Provider<'probe-api'>
    runtime.registry.set('pooled', foreign)

    await runtime.shutdown()

    expect(runtime.registry.get('pooled')).toBe(foreign)
    expect(runtime.unregisterCalls).not.toContain('pooled')
  })

  it('unregisters its own provider object on shutdown', async () => {
    const runtime = await launch()
    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.start()
    expect(runtime.registry.get('pooled')).toBeDefined()

    await runtime.shutdown()

    expect(runtime.registry.has('pooled')).toBe(false)
    expect(runtime.unregisterCalls).toContain('pooled')
  })

  it('re-registers the same provider object when the registry entry vanished', async () => {
    const runtime = await launch()
    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.start()
    const ours = runtime.registry.get('pooled')
    expect(ours).toBeDefined()

    // A sibling session's shutdown yanked the shared registry entry.
    runtime.registry.delete('pooled')
    runtime.registerCalls.length = 0

    await runtime.beforeAgentStart()

    expect(runtime.registry.get('pooled')).toBe(ours)
    expect(runtime.registerCalls).toContain('pooled')
  })

  it('does not re-register when its own object is still installed', async () => {
    const runtime = await launch()
    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.start()
    const ours = runtime.registry.get('pooled')
    runtime.registerCalls.length = 0

    await runtime.beforeAgentStart()

    expect(runtime.registry.get('pooled')).toBe(ours)
    expect(runtime.registerCalls).not.toContain('pooled')
  })
})
