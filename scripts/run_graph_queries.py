import time

from neurograph.config import git_hash
from neurograph.graph.client import get_driver, run
from neurograph.graph.queries import QUERIES


def main() -> None:
    t0 = time.perf_counter()
    driver = get_driver()
    print(f"== step 5.2 ten Cypher queries | commit {git_hash()} ==")
    for q in QUERIES:
        t = time.perf_counter()
        rows = run(driver, q["cypher"])
        print(f"\n{q['title']}  ({len(rows)} rows, {1000 * (time.perf_counter() - t):.0f} ms)")
        print(f"  expect: {q['expect']}")
        if rows:
            cols = list(rows[0])
            print("  " + " | ".join(cols))
            for r in rows[:10]:
                print("  " + " | ".join(str(r[c]) for c in cols))
    driver.close()
    print(f"\ntime {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
