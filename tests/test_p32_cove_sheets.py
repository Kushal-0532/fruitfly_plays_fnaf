import csv
import json

from PIL import Image

from scripts import cove_sheets


def test_sheets_and_index(tmp_path):
    (tmp_path / "frames").mkdir()
    for i in range(80):
        Image.new("RGB", (200, 100), (i, 0, 0)).save(tmp_path / "frames" / f"{i:06d}.png")
    (tmp_path / "meta.json").write_text(json.dumps({"times": [i / 10 for i in range(80)]}))
    rows = [{"t": i / 10, "monitor_up": 1 <= i % 40 < 30, "cam": "1C" if 1 <= i % 40 < 30 else None} for i in range(80)]
    log = tmp_path / "run.jsonl"
    log.write_text(json.dumps({"header": {}}) + "\n" + "\n".join(map(json.dumps, rows)) + "\n")
    n = cove_sheets.make(tmp_path, log)
    assert n == 12  # two looks, each capped at 6 tiles
    idx = list(csv.DictReader(open(tmp_path / "cove_index.csv")))
    assert len(idx) == 12 and {r["look_id"] for r in idx} == {"0", "1"}
    im = Image.open(tmp_path / "cove_sheets" / "sheet_00.png")
    assert im.width == cove_sheets.COLS * cove_sheets.TW
