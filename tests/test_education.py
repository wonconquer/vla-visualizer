"""Check saved-capture mapping and execution isolation, independently of browser acceptance."""

from copy import deepcopy
import json
from time import monotonic

import pytest
from streamlit.testing.v1 import AppTest
import torch

from vla_visualizer.config import ROOT
from vla_visualizer.adapters.education_catalog import INSTRUCTION_EXAMPLES
from vla_visualizer.execution.jobs import JobConfig
from vla_visualizer.ui.education_views import STAGES, norm_grid_html, pipeline_visuals
from vla_visualizer.ui.workflow import saved_capture, advance


def test_instruction_contract_and_compatibility():
    for _, text in INSTRUCTION_EXAMPLES:
        assert JobConfig(instruction=text).instruction == text
    with pytest.raises(ValueError):
        JobConfig(instruction='an unsupported task')
    with pytest.raises(ValueError):
        JobConfig(kind='training', instruction=INSTRUCTION_EXAMPLES[1][1])


def test_real_grid_geometry_and_no_fabrication():
    data = saved_capture().read()
    camera = next(iter(data['view_spec']['camera_to_native']))
    layout = data['capture']['intermediates']['vision'][camera]
    for stage, count in [('vision', 1024), ('connector', 64)]:
        record = next(r for r in data['capture']['intermediates']['stages'] if r['stage_name']==stage and r['context']['camera']==camera)
        summary = record['outputs']['tensor']
        html = norm_grid_html(summary, layout, connector=stage=='connector')
        assert html.count('<span title="token') == count
        broken = deepcopy(summary)
        broken['token_indices'] = broken['token_indices'][1:]
        assert norm_grid_html(broken, layout, connector=stage=='connector') is None
    assert pipeline_visuals(None,'Example')['instruction']['text']=='Example'
    assert not any('norms' in item for item in pipeline_visuals(None,'Example').values())
    assert pipeline_visuals(data,'different pending instruction')['instruction']['text']==data['manifest']['instruction']
    actual = next(r for r in data['capture']['intermediates']['stages']
                  if r['stage_name']=='action_velocity' and r['context']['flow_step_index']==4)
    assert pipeline_visuals(data,'',4)['velocity']['norms']==actual['outputs']['tensor']['token_norms']


def test_educational_replay_clock():
    assert advance(4,0,0,13,True,1)==(1,True,4)
    assert advance(4,0,0,13,False,1)==(0,False,0)
    assert advance(4,0,12,13,True,1)==(12,False,4)


def test_all_real_capture_views_and_play_without_inference(monkeypatch):
    from vla_visualizer.execution.jobs import JobManager
    from vla_visualizer.inference.runner import InferenceRunner
    def forbidden(*args, **kwargs):
        raise AssertionError('replay/selection attempted inference')
    monkeypatch.setattr(JobManager,'submit',forbidden)
    monkeypatch.setattr(InferenceRunner,'run_inference',forbidden)
    monkeypatch.setattr(InferenceRunner,'load',forbidden)
    app = AppTest.from_file(ROOT/'app.py', default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key='education_model').options==['SmolVLA']
    assert app.selectbox(key='instruction_example')
    app.button(key='saved_evidence').click().run()
    assert not app.exception
    snapshot=app.session_state['education_run'].snapshot
    for stage in STAGES:
        app.selectbox(key='_edu_stage').select(stage).run()
        assert not app.exception, stage
    app.selectbox(key='_edu_flow').select(4).run()
    app.slider(key='_edu_action').set_value(49).run()
    app.button(key='edu_reset').click().run()
    app.button(key='edu_play').click().run()
    app.session_state['edu_tick']=monotonic()-3
    app.run()
    assert app.session_state['edu_stage']=='top_connector'
    app.button(key='edu_pause').click().run()
    assert not app.session_state['edu_playing']
    app.selectbox(key='instruction_example').select(INSTRUCTION_EXAMPLES[1][1]).run()
    assert app.session_state['education_run'].snapshot==snapshot
    assert any('This capture used:' in e.value for e in app.warning)
    app.selectbox(key='workflow_mode').select('Simulation').run()
    assert app.button(key='start_job').disabled
    assert not app.get('imgs')
    assert app.selectbox(key='education_model').value=='smolvla'


def test_training_measured_events_and_fixed_probe():
    record=json.loads((ROOT/'assets/education/training_capture.json').read_text())
    assert record['visualization']['hook_count_after_step']==0
    assert 'target_velocity' in record['visualization']['objective']
    steps=record['visualization']['steps']
    assert len(steps)==2
    assert len({s['fixed_head_probe']['input_sha256'] for s in steps})==1
    for step in steps:
        assert step['forward']['loss_gradient_at_velocity']['l2_norm']>0
        assert step['fixed_head_probe']['max_abs_output_change']>0
        assert not step['fixed_head_probe']['rng_consumed']
        assert all(v['max_abs']>0 for v in step['parameter_update'].values())


def test_probe_does_not_change_rng_flags_or_parameters():
    from vla_visualizer.execution.training_capture import linear_probe
    head=torch.nn.Linear(4,2).train()
    hidden=torch.ones(1,3,4)
    before=torch.get_rng_state().clone()
    parameters={n:p.detach().clone() for n,p in head.named_parameters()}
    value=linear_probe(head,hidden)
    assert value.shape==(1,3,2) and not value.requires_grad
    assert head.training and torch.equal(torch.get_rng_state(),before)
    assert all(torch.equal(parameters[n],p) for n,p in head.named_parameters())


def test_training_ui_has_frozen_scope_and_update_views():
    app=AppTest.from_file(ROOT/'app.py',default_timeout=30).run()
    app.selectbox(key='workflow_mode').select('Training').run()
    app.button(key='saved_training').click().run()
    for phase in ['forward','objective','loss','backward','update','next']:
        app.selectbox(key='_edu_stage').select(phase).run()
        assert not app.exception,phase
    app.selectbox(key='_edu_stage').select('update').run()
    assert any('max |Δparameter|' in m.label for m in app.metric)
    app.selectbox(key='_training_step_view').select(1).run()
    assert not app.exception
