/** Per-browser UI preferences (localStorage). */
export type Prefs = {
  /** Play the spoken summary automatically after each anonymization. */
  autoplayVoice: boolean;
};

const KEY = "safeocr.prefs.v1";
const DEFAULTS: Prefs = { autoplayVoice: false };

export function loadPrefs(): Prefs {
  try {
    return { ...DEFAULTS, ...JSON.parse(window.localStorage.getItem(KEY) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

export function savePrefs(prefs: Prefs): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(prefs));
  } catch {
    /* storage unavailable */
  }
}
