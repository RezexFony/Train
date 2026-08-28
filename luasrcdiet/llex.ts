// llex.ts
// Lua 5.1+ lexical analyzer — direct port of LuaSrcDiet's llex.lua.
// Uses 1-based indexing helpers (sub/charAt) to mirror the original
// Lua string.sub/string.find semantics exactly, minimizing translation risk.

export type Token = string;
export type SemInfo = string;

const KEYWORDS = new Set([
  "and", "break", "do", "else", "elseif", "end", "false", "for", "function",
  "if", "in", "local", "nil", "not", "or", "repeat", "return", "then",
  "true", "until", "while",
]);

// Lua %p (ispunct): printable, not alnum, not space
const PUNCT_RE = /[!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/;
const ROUTE_STR = "-[\"'.=<>~"; // 0-based positions used for two-level routing

// --- 1-based string helpers, mirroring Lua semantics ---
function charAt(s: string, i: number): string {
  return i >= 1 && i <= s.length ? s[i - 1] : "";
}
function sub(s: string, a: number, b?: number): string {
  if (b === undefined) b = s.length;
  if (a < 1) a = 1;
  if (b > s.length) b = s.length;
  if (a > b) return "";
  return s.substring(a - 1, b);
}
function isDigit(c: string): boolean {
  return c >= "0" && c <= "9";
}
function isIdentStart(c: string): boolean {
  return /[A-Za-z_]/.test(c);
}
function isIdentPart(c: string): boolean {
  return /[A-Za-z0-9_]/.test(c);
}

function luaToNumber(v: string): number | null {
  if (/^0[xX][0-9a-fA-F]+$/.test(v)) return parseInt(v, 16);
  if (/^[0-9]*\.?[0-9]*([eE][+-]?[0-9]+)?$/.test(v) && /[0-9]/.test(v)) {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

export class LuaLexError extends Error {}

export interface LexResult {
  toklist: Token[];
  seminfolist: SemInfo[];
  toklnlist: number[];
}

export function lex(source: string, sourceName?: string): LexResult {
  // --- module-level mutable state, scoped per call (matches Lua's single-use module locals) ---
  const z = source;
  const sourceid = sourceName;
  let I = 1;
  let buff = 0;
  let ln = 1;
  const tok: Token[] = [];
  const seminfo: SemInfo[] = [];
  const tokln: number[] = [];

  function addtoken(token: Token, info: SemInfo): void {
    tok.push(token);
    seminfo.push(info);
    tokln.push(ln);
  }

  function chunkid(): string {
    if (sourceid && /^[=@]/.test(sourceid)) return sub(sourceid, 2);
    return "[string]";
  }

  function errorline(s: string, line?: number): never {
    throw new LuaLexError(`${chunkid()}:${line ?? ln}: ${s}`);
  }

  function inclinenumber(i: number, isTok: boolean): number {
    const old0 = charAt(z, i);
    i = i + 1;
    let old = old0;
    const c = charAt(z, i);
    if ((c === "\n" || c === "\r") && c !== old) {
      i = i + 1;
      old = old + c;
    }
    if (isTok) addtoken("TK_EOL", old);
    ln = ln + 1;
    I = i;
    return i;
  }

  function skip_sep(i: number): number {
    const s = charAt(z, i);
    i = i + 1;
    let count = 0;
    while (charAt(z, i + count) === "=") count++;
    i = i + count;
    I = i;
    return charAt(z, i) === s ? count : -count - 1;
  }

  function findFirstOf(chars: string, i: number): { p: number; r: string } | null {
    for (let j = i; j <= z.length; j++) {
      const c = charAt(z, j);
      if (chars.indexOf(c) !== -1) return { p: j, r: c };
    }
    return null;
  }

  function read_long_string(isStr: boolean, sep: number): string {
    let i = I + 1; // skip 2nd '['
    const c = charAt(z, i);
    if (c === "\r" || c === "\n") {
      i = inclinenumber(i, false);
    }
    while (true) {
      const found = findFirstOf("\r\n]", i);
      if (!found) {
        errorline(isStr ? "unfinished long string" : "unfinished long comment");
      }
      i = found.p;
      if (found.r === "]") {
        if (skip_sep(i) === sep) {
          const result = sub(z, buff, I);
          I = I + 1; // skip 2nd ']'
          return result;
        }
        i = I;
      } else {
        buff2Append("\n"); // handled via closure below
        i = inclinenumber(i, false);
      }
    }
  }

  // buff is tracked as a start index (like Lua's `buff` var reused for both
  // "start position" and "accumulated string" depending on context) — the
  // original code reuses `buff` as a position marker before read_* calls
  // and reassigns it to an accumulated string mid read_long_string. We
  // replicate that dual use with a small helper + local override.
  let buffAccum: string | null = null;
  function buff2Append(s: string): void {
    if (buffAccum === null) buffAccum = sub(z, buff, I);
    buffAccum += s;
    // re-anchor buff so a later `sub(z, buff, I)` isn't used (we return buffAccum directly instead)
  }

  function read_string(del: string): string {
    let i = I;
    while (true) {
      const found = findFirstOf("\n\r\\\"'", i);
      if (found) {
        let r = found.r;
        if (r === "\n" || r === "\r") {
          errorline("unfinished string");
        }
        i = found.p;
        if (r === "\\") {
          i = i + 1;
          r = charAt(z, i);
          if (r === "") break; // EOZ error
          const specialIdx = "abfnrtv\n\r".indexOf(r);
          if (specialIdx !== -1) {
            if (specialIdx >= 7) {
              i = inclinenumber(i, false);
            } else {
              i = i + 1;
            }
          } else if (!isDigit(r)) {
            // %D — any non-digit
            i = i + 1;
          } else {
            // \xxx sequence — 1 to 3 digits
            let j = i;
            let count = 0;
            while (count < 3 && isDigit(charAt(z, j))) {
              j++;
              count++;
            }
            const q = j - 1;
            const sVal = sub(z, i, q);
            i = q + 1;
            if (parseInt(sVal, 10) + 1 > 256) {
              errorline("escape sequence too large");
            }
          }
        } else {
          i = i + 1;
          if (r === del) {
            I = i;
            return sub(z, buff, i - 1);
          }
        }
      } else {
        break;
      }
    }
    errorline("unfinished string");
  }

  // --- init (shbang handling) ---
  {
    if (charAt(z, 1) === "#") {
      let j = 2;
      while (j <= z.length && charAt(z, j) !== "\r" && charAt(z, j) !== "\n") j++;
      const qPart = sub(z, 1, j - 1);
      I = I + qPart.length;
      addtoken("TK_COMMENT", qPart);
      let rPart = "";
      let k = j;
      if (charAt(z, k) === "\r") {
        rPart += "\r";
        k++;
      }
      if (charAt(z, k) === "\n") {
        rPart += "\n";
      }
      if (rPart.length > 0) inclinenumber(I, true);
    }
  }

  // --- main scan loop ---
  outer: while (true) {
    const i0 = I;

    // identifier / keyword
    if (isIdentStart(charAt(z, i0))) {
      let j = i0 + 1;
      while (isIdentPart(charAt(z, j))) j++;
      const r = sub(z, i0, j - 1);
      I = i0 + r.length;
      if (KEYWORDS.has(r)) addtoken("TK_KEYWORD", r);
      else addtoken("TK_NAME", r);
      continue outer;
    }

    // number
    {
      const hasDot = charAt(z, i0) === ".";
      const digitPos = hasDot ? i0 + 1 : i0;
      if (isDigit(charAt(z, digitPos))) {
        const p = i0;
        let i = hasDot ? i0 + 1 : i0;
        let j = i;
        while (charAt(z, j) === "." || isDigit(charAt(z, j))) j++;
        let r = "";
        if (charAt(z, j) === "e" || charAt(z, j) === "E") {
          r = charAt(z, j);
          j++;
        }
        let q = j - 1;
        i = q + 1;
        if (r.length === 1) {
          if (charAt(z, i) === "+" || charAt(z, i) === "-") i = i + 1;
        }
        let k = i;
        while (isIdentPart(charAt(z, k))) k++;
        q = k - 1;
        I = q + 1;
        const v = sub(z, p, q);
        if (luaToNumber(v) === null) {
          errorline("malformed number");
        }
        addtoken("TK_NUMBER", v);
        continue outer;
      }
    }

    // whitespace / newline
    {
      const t = charAt(z, i0);
      if (/\s/.test(t) && t !== "") {
        let j = i0 + 1;
        while (" \t\v\f".indexOf(charAt(z, j)) !== -1) j++;
        const r = sub(z, i0, j - 1);
        if (t === "\n" || t === "\r") {
          inclinenumber(i0, true);
        } else {
          I = j;
          addtoken("TK_SPACE", r);
        }
        continue outer;
      }
    }

    // '::' (goto label)
    if (charAt(z, i0) === ":" && charAt(z, i0 + 1) === ":") {
      I = i0 + 2;
      addtoken("TK_OP", "::");
      continue outer;
    }

    // punctuation / symbols
    {
      const r0 = charAt(z, i0);
      if (PUNCT_RE.test(r0)) {
        buff = i0;
        buffAccum = null;
        const idx = ROUTE_STR.indexOf(r0);
        let handled = false;

        if (idx !== -1) {
          if (idx <= 1) {
            if (idx === 0) {
              // '-' : possible comment
              const c = charAt(z, i0) + charAt(z, i0 + 1);
              if (c === "--") {
                const bracket = charAt(z, i0 + 2);
                let i = i0 + 2;
                let sep = -1;
                if (bracket === "[") {
                  sep = skip_sep(i);
                }
                if (sep >= 0) {
                  addtoken("TK_LCOMMENT", read_long_string(false, sep));
                } else {
                  const nl = findFirstOf("\n\r", i);
                  I = nl ? nl.p : z.length + 1;
                  addtoken("TK_COMMENT", sub(z, buff, I - 1));
                }
                handled = true;
              }
              // else fall through for lone "-"
            } else {
              // '[' — possible long string
              const sep = skip_sep(i0);
              if (sep >= 0) {
                addtoken("TK_LSTRING", read_long_string(true, sep));
              } else if (sep === -1) {
                addtoken("TK_OP", "[");
              } else {
                errorline("invalid long string delimiter");
              }
              handled = true;
            }
          } else if (idx <= 4) {
            if (idx <= 3) {
              // string delimiter: " or '
              I = i0 + 1;
              addtoken("TK_STRING", read_string(r0));
              handled = true;
            }
            // idx === 4 ('.') falls through to dots
          }
          // idx >= 5 falls through to relational
        }

        if (!handled) {
          let r = r0;
          if (idx !== -1 && idx === 4) {
            // dots: .|..|...
            let j = i0 + 1;
            if (charAt(z, j) === ".") {
              j++;
              if (charAt(z, j) === ".") j++;
            }
            r = sub(z, i0, j - 1);
          } else if (idx !== -1 && idx >= 5) {
            // relational: %p=?
            let j = i0 + 1;
            if (charAt(z, j) === "=") j++;
            r = sub(z, i0, j - 1);
          }
          I = i0 + r.length;
          addtoken("TK_OP", r);
        }
        continue outer;
      }
    }

    // any other single char, or end of stream
    {
      const r = charAt(z, i0);
      if (r !== "") {
        I = i0 + 1;
        addtoken("TK_OP", r);
        continue outer;
      }
      addtoken("TK_EOS", "");
      return { toklist: tok, seminfolist: seminfo, toklnlist: tokln };
    }
  }
}
