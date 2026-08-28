// init.ts
// LuaSrcDiet API — orchestrates lex -> parse -> optparser -> optlex -> (equiv check)
// NOTE: imports below reference llex.ts, lparser.ts, optlex.ts, optparser.ts,
// equiv.ts — not ported yet. This file won't compile until those land
// (next batches, in that order). Wiring it now so the shape is locked in.

import * as equiv from "./equiv";
import * as llex from "./llex";
import * as lparser from "./lparser";
import * as optlex from "./optlex";
import * as optparser from "./optparser";
import { merge } from "./utils";

export type OptFlags = Record<string, boolean>;

function optsToLegacy(opts: OptFlags): Record<string, boolean> {
  const res: Record<string, boolean> = {};
  for (const key in opts) {
    res[`opt-${key}`] = opts[key];
  }
  return res;
}

export const NONE_OPTS: OptFlags = {
  binequiv: false,
  comments: false,
  emptylines: false,
  entropy: false,
  eols: false,
  experimental: false,
  locals: false,
  numbers: false,
  srcequiv: false,
  strings: false,
  whitespace: false,
};

export const BASIC_OPTS: OptFlags = {
  ...NONE_OPTS,
  comments: true,
  emptylines: true,
  srcequiv: true,
  whitespace: true,
};

export const DEFAULT_OPTS: OptFlags = {
  ...BASIC_OPTS,
  locals: true,
  numbers: true,
};

export const MAXIMUM_OPTS: OptFlags = {
  ...DEFAULT_OPTS,
  entropy: true,
  eols: true,
  strings: false,   // disabled, matches original override
  srcequiv: false,  // disabled for non-standard Lua syntax (e.g. compound operators)
};

export function optimize(opts: OptFlags | null, source: string): string {
  if (typeof source !== "string") {
    throw new Error(`bad argument #2: expected string, got a ${typeof source}`);
  }

  const finalOpts = opts ? (merge(NONE_OPTS, opts) as OptFlags) : DEFAULT_OPTS;
  const legacyOpts = optsToLegacy(finalOpts);

  const { toklist, seminfolist, toklnlist } = llex.lex(source);
  const xinfo = lparser.parse(toklist, seminfolist, toklnlist);

  optparser.optimize(legacyOpts, toklist, seminfolist, xinfo);

  const warn = optlex.warn;
  const optimResult = optlex.optimize(legacyOpts, toklist, seminfolist, toklnlist);
  const optimSource = optimResult.seminfolist.join("");

  if (finalOpts.srcequiv && !finalOpts.experimental) {
    equiv.init(legacyOpts, llex, warn);
    equiv.source(source, optimSource);

    if (warn.SRC_EQUIV) {
      throw new Error("Source equivalence test failed!");
    }
  }

  return optimSource.replace(/\n/g, " ");
}
