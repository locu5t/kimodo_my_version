import json
from decimal import Decimal
import pytest
from kimodo_motion_studio.timeline import parse_prompts, frame_prompts, export_prompts, load_json, viser_bounds

FORMATS = [
    [{"text":"strut", "duration":6}, {"text":"wave", "duration":2.5}],
    {"prompts":[{"text":"strut", "duration_seconds":"6"}, {"text":"wave", "start_seconds":6, "duration_seconds":2.5}]},
    {"texts":["strut", "wave"], "durations":[6, 2.5]},
]

@pytest.mark.parametrize("data", FORMATS)
def test_formats(data):
    p = parse_prompts(data)
    assert [x.duration for x in p] == [Decimal("6"), Decimal("2.5")]
    assert [(x.start,x.end) for x in frame_prompts(p,30)] == [(0,180),(180,255)]
    assert [(x.start,x.end) for x in frame_prompts(p,24,1)] == [(1,145),(145,205)]

@pytest.mark.parametrize("fps", [24,25,30,60,23.976,29.97])
def test_roundtrip_native(fps):
    p = parse_prompts(FORMATS[0])
    for native in (True, False):
        assert parse_prompts(export_prompts(p,native)) == p
        assert all(x.count > 0 for x in frame_prompts(p,fps))

def test_shared_viser_final_boundary():
    assert viser_bounds(parse_prompts(FORMATS[0]),30) == [("strut",0,180),("wave",180,254)]

def test_cumulative_rounding_not_per_segment():
    p = parse_prompts([{"text":"walk", "duration":0.15}]*10)
    spans = frame_prompts(p,30)
    assert spans[-1].end == 45
    assert [s.count for s in spans][:4] == [5,4,5,4]

@pytest.mark.parametrize("duration", [0,-2,"NaN",float("inf"),True,None,"six seconds",{},[]])
def test_bad_duration(duration):
    with pytest.raises(ValueError):
        parse_prompts({"text":"walk", "duration":duration})

@pytest.mark.parametrize("data", [[],{},None,"walk", {"texts":["a"],"durations":[]},
    [{"text":"", "duration":1}], [{"text":5,"duration":1}],
    [{"text":"a","duration":1,"start":2}],
    [{"text":"a","duration":1},{"text":"b","duration":2,"start":0.5}],
    {"schema_version":2,"prompts":[{"text":"a","duration":1}]},
    {"prompts":[],"texts":[]}, [{"text":"a","duration":1,"duration_seconds":1}],
    [{"text":"a","duration":1,"start":0,"start_seconds":0}],
    [{"text":"a","duration":1201}],
    [{"text":"a","duration":1}]*257])
def test_bad_input(data):
    with pytest.raises(ValueError):
        parse_prompts(data)

def test_subframe_rejected():
    with pytest.raises(ValueError):
        frame_prompts(parse_prompts({"text":"walk","duration":0.001}),30)

@pytest.mark.parametrize("fps", [0,-1,float("nan"),float("inf"),True])
def test_bad_fps(fps):
    with pytest.raises(ValueError):
        frame_prompts(parse_prompts(FORMATS[0]),fps)

def test_bom_json_and_nonfinite(tmp_path):
    file = tmp_path / "prompts.json"
    file.write_text(json.dumps(FORMATS[0]),encoding="utf-8-sig")
    assert parse_prompts(load_json(file))[0].text == "strut"
    file.write_text('{"text":"walk","duration":NaN}')
    with pytest.raises(ValueError):
        load_json(file)
