"""Exercise the educational UI in a real browser and save screenshots and video."""

import argparse
import asyncio
import json
from pathlib import Path


async def verify(url, output):
    from playwright.async_api import async_playwright

    shots = []
    async with async_playwright() as engine:
        browser = await engine.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
            record_video_dir=str(output / "recordings"),
            record_video_size={"width": 1440, "height": 1000},
        )
        await page.goto(url)
        await page.get_by_role("heading", name="SmolVLA, from inputs to actions").wait_for()
        await page.frame_locator('iframe[title*="smolvla_pipeline"]').locator("svg").wait_for()

        async def shot(name):
            path = output / (name + ".png")
            await page.screenshot(path=str(path), full_page=True)
            shots.append(str(path))

        async def choose(label, text):
            await page.get_by_role("combobox", name=label, exact=True).click()
            await page.get_by_role("option", name=text, exact=True).click()

        await shot("01_initial")
        await page.get_by_role("button", name="Open recorded 10-iteration loop", exact=True).click()
        await choose("Current exploration stage", "Current action sample xₜ")
        await page.get_by_role("button", name="Play", exact=True).click()
        await page.wait_for_timeout(12500)
        await page.get_by_role("button", name="Pause", exact=True).click()
        await shot("02_pipeline_replay")
        frame = page.frame_locator('iframe[title*="smolvla_pipeline"]')
        await frame.locator('.glyph[data-key="vision_top"]').first.hover()
        await frame.locator("#hover-preview").wait_for(state="visible")
        assert not await frame.locator("#inspector").is_visible()
        await shot("03_vision_hover")
        await frame.locator('.glyph[data-key="vision_top"]').first.click()
        await frame.locator("#token-index").fill("350")
        await frame.locator("#token-index").press("Tab")
        await shot("03_vision_pinned_token")
        await frame.locator("#unpin").click()
        # Keyboard activation pins the same CPU artifact without changing the stage.
        await frame.locator('.glyph[data-key="vision_top"]').first.focus()
        await page.keyboard.press("Enter")
        await frame.locator("#inspector").wait_for(state="visible")
        await page.keyboard.press("Escape")
        assert not await frame.locator("#inspector").is_visible()
        await frame.locator('g[data-stage="top_connector"]').click()
        await page.get_by_role("heading", name="Top · visual connector", exact=True).wait_for()
        await shot("03_intermediate")
        await choose("Scenario / environment", "SO100 · dataset episode/frame exploration")
        slider = page.get_by_role("slider", name="Recorded frame")
        await slider.focus()
        await page.keyboard.press("ArrowRight")
        await shot("06_dataset_other_frame")
        await choose("Instruction example", "Sentence variation 3 · task success not verified")
        await shot("07_instruction_example")
        await choose("Mode", "Training")
        await page.get_by_role("button", name="Open recorded training experiment", exact=True).click()
        await choose("Current training stage", "Optimizer update")
        await shot("04_training_update")
        await choose("Mode", "Simulation")
        assert await page.get_by_role("button", name="Start simulation").is_disabled()
        await shot("05_simulation_status")
        await browser.close()
    return shots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase5_education/screenshots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        screenshots = asyncio.run(verify(args.url, args.output))
        result = {
            "status": "PASS_BROWSER_AUTOMATION",
            "screenshots": screenshots,
            "note_ko": "Real browser interaction and capture. Review screenshots separately for clipping and readability.",
        }
    except Exception as exc:
        result = {
            "status": "BLOCKED",
            "screenshots": [],
            "type": type(exc).__name__,
            "error": str(exc),
        }
    (args.output.parent / "browser_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["screenshots"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
