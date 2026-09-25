// probe-v10.mjs — real-stack reproduction of "a few successes, then the next
// request dies with commandcode's 402 while both opencode-go keys are alive".
//
// Runs the fork's real service + lift + virtual pipeline over the real
// multiprovider-auth.json. opencode-go uses the real keys; commandcode is a
// stand-in that returns the exact observed failure (HTTP 402 "insufficient
// credits" after ~1.2s) so the probe never hammers the real endpoint.
//
// Modes (PROBE_MODE):
//   real  — real network: opencode-go at its real baseUrl, 8 short requests.
//   slow  — opencode-go pointed at a local server that answers *just after*
//           the 16s first-token watchdog (healthy backend, slow first byte).
//   stall — opencode-go pointed at a local server that accepts and never
//           answers (genuine silent stall).
//   mixed — local server: the first PROBE_FAST_HITS requests answer instantly
//           (so the session starts healthy), the rest answer after SLOW_MS —
//           models "a few successes, then every next request dies".
//
//   PROBE_MODE=real bun probe-v10.mjs      (MULTIPROVIDER_DEBUG=1 for trace)
import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import {
  createAssistantMessageEventStream,
  createProvider,
  normalizeContext,
} from '@earendil-works/pi-ai'
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy'
import {
  createVirtualIntegrations,
  createVirtualProvider,
  liftProvider,
  MultiProviderService,
} from './src/index.ts'

const AUTH_PATH = process.env.MP_AUTH ?? '/root/.pi/agent/multiprovider-auth.json'
const VIRTUAL_ID = 'modelpool'
const MODEL_ID = process.env.PROBE_MODEL ?? 'deepseek-v4.1-flash'
const MODE = process.env.PROBE_MODE ?? 'real'
const REQUESTS = Number(process.env.PROBE_REQUESTS ?? 8)
const SESSION_ID = process.env.PROBE_SESSION ?? 'probe-session-v10'
const SLOW_MS = Number(process.env.PROBE_SLOW_MS ?? 17000)
const FAST_HITS = Number(process.env.PROBE_FAST_HITS ?? 3)
const COMMANDCODE_DELAY_MS = 1200

const auth = JSON.parse(readFileSync(AUTH_PATH, 'utf8'))
const openPool = auth.providers['opencode-go']
const vconfig = auth.virtuals[VIRTUAL_ID]
const vmodel = vconfig.models.find(model => model.id === MODEL_ID)
if (vmodel === undefined) throw new Error(`virtual model ${MODEL_ID} not found`)
const openBackend = vmodel.backends.find(backend => backend.providerId === 'opencode-go')
const ccBackend = vmodel.backends.find(backend => backend.providerId === 'commandcode')

const keysById = new Map(openPool.accounts.map(account => [account.id, account.credential.key]))
const short = id => (typeof id === 'string' ? id.slice(0, 8) : id)

function modelFrom(template, id, providerId, baseUrl) {
  return {
    id,
    name: id,
    api: template?.api ?? 'openai-completions',
    provider: providerId,
    baseUrl: baseUrl ?? template?.baseUrl ?? '',
    reasoning: template?.reasoning ?? false,
    input: template?.input ?? ['text'],
    cost: template?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: template?.contextWindow ?? 128_000,
    maxTokens: template?.maxTokens ?? 8_192,
  }
}

// ---- local opencode-go stand-in used by the slow/stall modes ---------------
let localServer
let localBaseUrl
let localHits = 0
function startLocalServer() {
  localServer = createServer((req, res) => {
    req.resume()
    if (MODE === 'stall') return // accept, never answer — genuine silent stall
    localHits += 1
    const delay = MODE === 'mixed' && localHits > FAST_HITS ? SLOW_MS : 0
    req.on('end', () => {
      setTimeout(() => {
        res.writeHead(200, { 'content-type': 'text/event-stream' })
        const base = { id: 'probe', object: 'chat.completion.chunk', created: Date.now(), model: MODEL_ID }
        res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: 'ok' }, finish_reason: null }] })}\n\n`)
        res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`)
        res.write('data: [DONE]\n\n')
        res.end()
      }, delay)
    })
  })
  return new Promise(resolve => {
    localServer.listen(0, '127.0.0.1', () => {
      localBaseUrl = `http://127.0.0.1:${localServer.address().port}/v1`
      resolve(localBaseUrl)
    })
  })
}

// ---- commandcode stand-in: 402 after ~1.2s, exactly as observed ------------
function commandcodeStandIn() {
  const model = modelFrom(ccBackend.template, ccBackend.modelId, 'commandcode', ccBackend.template.baseUrl)
  const fail = () => {
    const stream = createAssistantMessageEventStream()
    setTimeout(() => {
      const message = {
        role: 'assistant', content: [], api: model.api, provider: 'commandcode', model: model.id,
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: 'error', errorMessage: '402 insufficient credits', timestamp: Date.now(),
      }
      stream.push({ type: 'error', reason: 'error', error: message })
      stream.end(message)
    }, COMMANDCODE_DELAY_MS)
    return stream
  }
  return {
    id: 'commandcode', name: 'commandcode (probe stand-in)',
    baseUrl: ccBackend.template.baseUrl,
    auth: { apiKey: { name: 'probe', resolve: async () => ({ auth: {} }) } },
    getModels: () => [model],
    stream: fail,
    streamSimple: fail,
  }
}

async function main() {
  let openBaseUrl = openBackend.template.baseUrl
  if (MODE !== 'real') {
    const base = await startLocalServer()
    // openAI-completions appends /chat/completions to baseUrl.
    openBaseUrl = base
    console.error(`[probe] MODE=${MODE} opencode-go -> ${openBaseUrl} (slow=${SLOW_MS}ms, fastHits=${FAST_HITS})`)
  } else {
    console.error(`[probe] MODE=real opencode-go -> ${openBaseUrl}`)
  }

  const service = new MultiProviderService()

  const api = openAICompletionsApi()
  const openBase = createProvider({
    id: 'opencode-go',
    name: 'opencode-go',
    baseUrl: openBaseUrl,
    auth: { apiKey: { name: 'probe', resolve: async () => ({ auth: {} }) } },
    models: [modelFrom(openBackend.template, openBackend.modelId, 'opencode-go', openBaseUrl)],
    api,
  })

  const openIntegration = {
    id: 'opencode-go',
    label: 'opencode-go',
    accounts: () => openPool.accounts
      .filter(account => account.enabled !== false)
      .map(account => ({
        id: account.id,
        label: account.label,
        authKind: 'api-key',
        credentialRef: account.id,
        metadata: {},
      })),
    affinityKey: () => SESSION_ID,
    resolveAuth: async account => ({
      auth: { apiKey: keysById.get(account.id) },
      source: 'probe stored account',
    }),
    onFailover: () => false,
  }
  service.registerProvider(openIntegration)
  service.updatePool('opencode-go', { policy: openPool.policy, affinity: openPool.affinity })

  const openLifted = liftProvider(openBase, service, {
    ...openIntegration,
    affinityKey: () => SESSION_ID,
    onFailover: () => false,
  })

  const commandcode = commandcodeStandIn()

  for (const integration of createVirtualIntegrations(vconfig, {})) {
    service.registerProvider(integration)
  }

  const virtual = createVirtualProvider({
    service,
    config: vconfig,
    onFailover: () => false,
    getAffinityKey: () => SESSION_ID,
    getBackingProvider: providerId => {
      if (providerId === 'opencode-go') return openLifted
      if (providerId === 'commandcode') return commandcode
      return undefined
    },
    resolveAmbientAuth: async () => ({ ok: false, error: 'probe: no ambient auth' }),
  })

  const model = virtual.getModels().find(candidate => candidate.id === MODEL_ID)
  const context = normalizeContext({
    messages: [{ role: 'user', content: [{ type: 'text', text: 'reply with the single word ok' }] }],
  })

  for (let i = 1; i <= REQUESTS; i++) {
    const startedAt = Date.now()
    process.stderr.write(`\n=== REQUEST ${i} ===\n`)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(new Error('probe outer timeout')), 150_000)
    let outcome = 'no-terminal-event'
    let detail = ''
    try {
      const stream = virtual.streamSimple(model, context, {
        maxTokens: 32,
        signal: controller.signal,
        // opencode-go rejects requests without a routing session header
        // (HTTP 400 MissingSessionID); pi-web supplies this via its provider.
        headers: { 'x-opencode-session': SESSION_ID },
      })
      for await (const event of stream) {
        if (event.type === 'done') {
          outcome = 'success'
          detail = (event.message.content ?? []).map(part => part.type === 'text' ? part.text : '').join('').slice(0, 40)
        } else if (event.type === 'error') {
          outcome = 'error'
          detail = event.error?.errorMessage ?? event.reason ?? 'unknown'
        }
      }
    } catch (error) {
      outcome = 'throw'
      detail = error instanceof Error ? error.message : String(error)
    } finally {
      clearTimeout(timeout)
    }
    const snapshot = (await service.snapshot()).providers
    const line = snapshot.map(pool => `${pool.id}[${pool.accounts.map(account =>
      `${short(account.id)}:${account.status}${account.cooldownUntil ? '(cd ' + Math.max(0, Math.round((account.cooldownUntil - Date.now()) / 1000)) + 's,f' + account.consecutiveFailures + ')' : ''}`,
    ).join(' ')}]`).join('  ')
    console.error(`[probe] request ${i}: ${outcome} in ${Date.now() - startedAt}ms :: ${detail}`)
    console.error(`[probe] state: ${line}`)
  }

  if (MODE !== 'real' && typeof localServer !== 'undefined') localServer.close()
}

main().catch(error => {
  console.error('[probe] fatal', error)
  process.exitCode = 1
})
