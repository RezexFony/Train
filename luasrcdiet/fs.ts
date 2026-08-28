// fs.ts
// File system helpers. Kept for CLI/local-testing parity — the web
// flow won't use these (source comes in/out as a string over HTTP,
// handled by Node's own fs or straight from the request body).
import * as fs from "fs";

function normalizeIoError(name: string, err: string): string {
  const prefix = `${name}: `;
  return err.startsWith(prefix) ? err.slice(prefix.length) : err;
}

export function readFile(filename: string): { content?: string; error?: string } {
  try {
    let content = fs.readFileSync(filename, "utf8");
    const UTF8_BOM = "\u00EF\u00BB\u00BF";
    if (content.startsWith(UTF8_BOM)) {
      content = content.slice(UTF8_BOM.length);
    }
    return { content };
  } catch (e) {
    return { error: `Could not open ${filename} for reading: ${normalizeIoError(filename, String(e))}` };
  }
}

export function writeFile(filename: string, data: string): { ok?: true; error?: string } {
  try {
    fs.writeFileSync(filename, data, "utf8");
    return { ok: true };
  } catch (e) {
    return { error: `Could not write ${filename}: ${normalizeIoError(filename, String(e))}` };
  }
  }
