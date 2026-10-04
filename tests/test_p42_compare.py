import compare_policies as C


def test_boot_and_tiny_compare(tmp_path):
    lo, hi = C.boot([1, 2, 3, 4, 5], n=200)
    assert 1 <= lo <= 3 <= hi <= 5
    C.main(["--nights", "2", "--seeds", "2", "--max-s", "20", "--out", str(tmp_path)])
    import json
    rep = json.loads((tmp_path / "compare.json").read_text())["report"]["2"]
    assert set(rep) == set(C.POLICIES) and rep["supervisor"]["paired_diff_vs_supervisor"] == 0
    assert (tmp_path / "table.md").exists() and (tmp_path / "survival.png").exists()
