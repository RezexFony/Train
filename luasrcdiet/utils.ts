// utils.ts
// General utility functions. Not part of public API.

export function merge(...tables: Record<string, unknown>[]): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const tab of tables) {
    for (const key in tab) {
      result[key] = tab[key];
    }
  }
  return result;
}
