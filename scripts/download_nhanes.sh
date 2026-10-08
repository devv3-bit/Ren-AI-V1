#!/usr/bin/env bash
# scripts/download_nhanes.sh
#
# Fetch the public CDC NHANES files used by Ren AI v2 (src/nhanes/).
#
# Cycles : 2005-06 (_D) 2007-08 (_E) 2009-10 (_F) 2011-12 (_G)
#          2013-14 (_H) 2015-16 (_I) 2017-Mar 2020 pre-pandemic (P_ prefix)
# Files  : DEMO BIOPRO ALB_CR CBC BPX DIQ BMX GHB  (P cycle: P_BPXO instead of BPX)
# Total  : 7 cycles x 8 files = 56 SAS transport (.xpt) files
#
# 2017-18 (_J) is deliberately NOT downloaded: it is contained in the P_ files.
#
# Behaviour:
#   * If a local copy exists in $NHANES_LOCAL_DIR (default ~/Desktop/nhanes_ckd)
#     it is copied instead of downloaded.
#   * Files that already exist (non-empty) in data/external/nhanes/raw are skipped,
#     so the script is safe to re-run.
#   * Exit status is non-zero if any file could not be obtained.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/external/nhanes/raw"
LOCAL_SRC="${NHANES_LOCAL_DIR:-$HOME/Desktop/nhanes_ckd}"
BASE_URL="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# "<cycle letter>:<year used in the CDC URL>"
CYCLES=("D:2005" "E:2007" "F:2009" "G:2011" "H:2013" "I:2015" "P:2017")
COMPONENTS=(DEMO BIOPRO ALB_CR CBC BPX DIQ BMX GHB)

mkdir -p "${RAW_DIR}"

# Map (cycle, component) -> CDC file basename without extension.
xpt_basename() {
  local cyc="$1" comp="$2"
  if [[ "${cyc}" == "P" ]]; then
    [[ "${comp}" == "BPX" ]] && comp="BPXO"
    printf 'P_%s' "${comp}"
  else
    printf '%s_%s' "${comp}" "${cyc}"
  fi
}

n_total=0
n_ok=0
n_fail=0
for entry in "${CYCLES[@]}"; do
  cyc="${entry%%:*}"
  year="${entry##*:}"
  for comp in "${COMPONENTS[@]}"; do
    n_total=$((n_total + 1))
    name="$(xpt_basename "${cyc}" "${comp}")"
    dest="${RAW_DIR}/${name}.xpt"

    if [[ -s "${dest}" ]]; then
      echo "exists    ${name}.xpt"
      n_ok=$((n_ok + 1))
      continue
    fi

    local_copy=""
    for cand in "${LOCAL_SRC}/${name}.xpt" "${LOCAL_SRC}/${name}.XPT"; do
      if [[ -s "${cand}" ]]; then
        local_copy="${cand}"
        break
      fi
    done
    if [[ -n "${local_copy}" ]]; then
      cp "${local_copy}" "${dest}"
      echo "copied    ${name}.xpt  (from ${local_copy})"
      n_ok=$((n_ok + 1))
      continue
    fi

    url="${BASE_URL}/${year}/DataFiles/${name}.xpt"
    if curl -fsSL --retry 3 --retry-delay 5 --max-time 600 -o "${dest}" "${url}"; then
      echo "download  ${name}.xpt  ($(du -h "${dest}" | cut -f1))"
      n_ok=$((n_ok + 1))
    else
      echo "FAILED    ${name}.xpt  (${url})" >&2
      rm -f "${dest}"
      n_fail=$((n_fail + 1))
    fi
  done
done

echo
echo "NHANES files: ${n_ok}/${n_total} available in ${RAW_DIR} (${n_fail} failed)"
[[ "${n_fail}" -eq 0 ]]
