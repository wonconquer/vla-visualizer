"""Verify selected instructions through actual tokenization, inference and isolated replay."""

from hashlib import sha256
import json
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

from vla_visualizer.config import ROOT
from vla_visualizer.adapters.education_catalog import INSTRUCTION_EXAMPLES
from vla_visualizer.execution.jobs import TERMINAL, read_json
from vla_visualizer.inference.replay import select_records


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('ECE525_EDUCATION_REAL')!='1',reason='Requires the pinned SmolVLA cache')
def test_selected_instruction_reaches_actual_model(monkeypatch):
    from vla_visualizer.ui.workflow import manager
    from vla_visualizer.adapters.smolvla import SmolVLAAdapter, BACKBONE, BACKBONE_REVISION, BACKBONE_FILES
    from vla_visualizer.contracts import LoadOptions
    from transformers import AutoTokenizer
    destination=ROOT/'outputs/instruction_checks/ui_jobs'
    monkeypatch.setenv('VLA_ARTIFACT_DIR',str(destination))
    manager.clear()
    app=AppTest.from_file(ROOT/'app.py',default_timeout=60).run()
    assert not app.exception
    instruction=INSTRUCTION_EXAMPLES[1][1]
    app.selectbox(key='instruction_example').select(instruction).run()
    app.button(key='start_job').click().run()
    assert not app.exception
    job=app.session_state['active_job']
    owner=app.session_state['job_owner']
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        status=manager().status(job,owner)
        if status['state'] in TERMINAL:
            break
        time.sleep(.5)
        app.run()
        assert not app.exception
    assert status['state']=='completed',status
    app.run()
    assert not app.exception
    directory=manager().directory(job)
    capture=read_json(directory/'result.json')
    task=select_records(capture,['language_tokenization'])[0]['context']['task']
    assert task==[instruction+'\n']
    processor_ids=select_records(capture,['language_tokenization'])[0]['outputs']['token_ids']['values']
    embedding_ids=select_records(capture,['language'])[0]['inputs']['tensor']['values']
    assert processor_ids==embedding_ids
    from vla_visualizer.config import cache_dir
    source=SmolVLAAdapter._snapshot(BACKBONE,BACKBONE_REVISION,BACKBONE_FILES,LoadOptions(cache_dir=str(cache_dir())))
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    expected=tokenizer(instruction+'\n',padding='max_length',truncation=True,max_length=len(processor_ids))['input_ids']
    assert processor_ids==expected
    assert capture['metadata']['native_action_shape']==[1,50,6]
    assert capture['intermediates']['hook_count_before']==capture['intermediates']['hook_count_after']==0
    count=len(list(destination.glob('*/status.json')))
    digest=sha256((directory/'replay.json').read_bytes()).hexdigest()
    for stage in ('top_vision','top_connector','wrist_vision','language_embedding','state_normalize','state_project','prefix','conditioning','expert','velocity','actions'):
        app.selectbox(key='_edu_stage').select(stage).run()
        assert not app.exception
    app.button(key='edu_reset').click().run()
    app.button(key='edu_play').click().run()
    app.session_state['edu_tick']=time.monotonic()-3
    app.run()
    assert app.session_state['edu_stage']=='top_connector'
    app.button(key='edu_pause').click().run()
    assert len(list(destination.glob('*/status.json')))==count
    assert sha256((directory/'replay.json').read_bytes()).hexdigest()==digest
    report={'status':'PASS','run_id':job,'instruction':instruction,'captured_task':task,
            'processor_ids':processor_ids,'model_embedding_input_ids':embedding_ids,'tokenizer_expected_ids':expected,
            'native_action_shape':capture['metadata']['native_action_shape'],
            'hook_count_after':capture['intermediates']['hook_count_after'],
            'replay_additional_jobs':0,'replay_sha256_unchanged':True,
            'browser_visual_acceptance':False,'scope':'Actual model inference and AppTest; browser screenshot acceptance is separate'}
    (ROOT/'outputs/instruction_checks/real_instruction_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
