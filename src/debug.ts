// Env-gated trace for the failover path. Off unless MULTIPROVIDER_DEBUG is
// set; writes one line per event to stderr so it never pollutes the model
// stream or the TUI. Reads the env on every call so a probe (or a live
// session) can flip it without a module reload.
export function debugEnabled(): boolean {
  const value = process.env.MULTIPROVIDER_DEBUG
  return value !== undefined && value !== '' && value !== '0' && value !== 'false'
}

export function debugLog(event: string, fields: Record<string, unknown> = {}): void {
  if (!debugEnabled()) return
  let line = 'mp ' + new Date().toISOString() + ' ' + event
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined) continue
    const rendered = typeof value === 'string' ? value : JSON.stringify(value)
    line += ' ' + key + '=' + rendered
  }
  process.stderr.write(line + '\n')
}

// Credentials must never reach the log; account ids and labels are safe
// (they are the pool's public handles).
export function shortId(id: string | undefined): string | undefined {
  return id === undefined ? undefined : id.slice(0, 8)
}
