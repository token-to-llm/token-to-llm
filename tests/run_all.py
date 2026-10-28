"""批量运行 src 下所有脚本，报告通过/失败。"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    scripts = sorted(glob.glob(os.path.join(ROOT, "src", "*", "*.py")))
    print(f"共 {len(scripts)} 个脚本\n")

    ok, bad = [], []
    for s in scripts:
        rel = os.path.relpath(s, ROOT)
        p = subprocess.run([sys.executable, s], capture_output=True, text=True, timeout=900)
        if p.returncode == 0:
            ok.append(rel)
            lines = [l for l in p.stdout.strip().split("\n") if l.strip()]
            print(f"[OK]   {rel}")
            for l in lines[-3:]:
                print(f"         {l}")
        else:
            bad.append(rel)
            print(f"[FAIL] {rel}")
            print(p.stderr[-2000:])

    print(f"\n通过 {len(ok)} / {len(scripts)}")
    if bad:
        print("失败清单:")
        for b in bad:
            print("  -", b)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
