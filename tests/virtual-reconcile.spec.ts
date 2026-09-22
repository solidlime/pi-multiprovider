import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createProvider, type Api, type Model, type Provider } from '@earendil-works/pi-ai'
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent'
import type { VirtualProviderConfig } from '../src/index.ts'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The extension resolves its auth store from the agent dir at load, so the
// harness points Pi's agent dir at a scratch directory before anything
// constructs a store.
const agentDir = mkdtempSync(join(tmpdir(), 'pi-multiprovider-virtual-'))
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

interface Notification {
  message: string
  level: string
}

interface ExtensionHarness {
  registered: Map<string, Provider<Api>>
  notifications: Notification[]
  ctx: ExtensionContext
  start(): Promise<void>
  beforeAgentStart(): Promise<void>
}

// Boots the real bundled extension against duck-typed Pi APIs, recording
// provider registrations so tests can assert what reached the registry.
async function launch(options: { failProviderIds?: readonly string[] } = {}): Promise<ExtensionHarness> {
  const entries: unknown[] = []
  const notifications: Notification[] = []
  const registered = new Map<string, Provider<Api>>()
  const failing = new Set(options.failProviderIds ?? [])
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => Promise<void> | void)[]>()
  const bus = new Map<string, Set<(value: unknown) => void>>()
  const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>()

  const ctx = {
    ui: {
      notify(message: string, level: string = 'info') { notifications.push({ message, level }) },
      async select() { return undefined },
      async input() { return undefined },
      async confirm() { return false },
      async custom() { return undefined },
    },
    mode: 'tui',
    hasUI: true,
    cwd: process.cwd(),
    sessionManager: { getSessionId: () => 'session-1', getEntries: () => entries },
    modelRegistry: {
      getProvider: (id: string) => registered.get(id) ?? (id === base.id ? base : undefined),
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
  } as unknown as ExtensionContext

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
    on(name: string, handler: () => Promise<void> | void) {
      const list = handlers.get(name) ?? []
      list.push(handler as (event: unknown, ctx: unknown) => Promise<void> | void)
      handlers.set(name, list)
    },
    registerProvider(provider: Provider<Api>) {
      if (failing.has(provider.id)) throw new Error(`provider id already registered: ${provider.id}`)
      registered.set(provider.id, provider)
    },
    unregisterProvider(providerId: string) { registered.delete(providerId) },
    getAllTools: () => [],
    registerCommand(name: string, def: { handler: (args: string, ctx: unknown) => Promise<void> }) {
      commands.set(name, def)
    },
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

  await multiprovider(pi as unknown as ExtensionAPI)

  return {
    registered,
    notifications,
    ctx,
    async start() {
      for (const handler of handlers.get('session_start') ?? []) {
        await handler({ type: 'session_start', reason: 'startup' }, ctx)
      }
    },
    async beforeAgentStart() {
      for (const handler of handlers.get('before_agent_start') ?? []) {
        await handler({ type: 'before_agent_start' }, ctx)
      }
    },
  }
}

beforeEach(async () => {
  for (const config of await store.listVirtualProviders()) {
    await store.removeVirtualProvider(config.id)
  }
})

describe('late virtual provider registration via reconcile', () => {
  it('registers a virtual provider added to the store after extension load on session_start', async () => {
    const runtime = await launch()
    // Extension load ran against a store with no virtual config.
    expect(runtime.registered.has('pooled')).toBe(false)

    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.start()

    expect(runtime.registered.get('pooled')?.getModels()).toEqual([
      expect.objectContaining({ id: 'ultra', provider: 'pooled' }),
    ])
  })

  it('registers a late-added virtual provider on before_agent_start', async () => {
    const runtime = await launch()
    expect(runtime.registered.has('pooled')).toBe(false)

    await store.saveVirtualProvider(virtualConfig('pooled'))
    await runtime.beforeAgentStart()

    expect(runtime.registered.get('pooled')?.getModels()).toEqual([
      expect.objectContaining({ id: 'ultra', provider: 'pooled' }),
    ])
  })

  it('keeps registering other virtual providers when one registration fails', async () => {
    const runtime = await launch({ failProviderIds: ['clash'] })
    await store.saveVirtualProvider(virtualConfig('clash'))
    await store.saveVirtualProvider(virtualConfig('kept'))
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    try {
      await runtime.start()
      expect(runtime.registered.has('kept')).toBe(true)
      expect(runtime.notifications.filter(item =>
        item.level === 'warning' && item.message.includes('multiprovider') && item.message.includes('"clash"'),
      )).not.toHaveLength(0)
      expect(errors.mock.calls.some(args => String(args[0]).includes('clash'))).toBe(true)
    } finally {
      errors.mockRestore()
    }
  })
})
