"""Timeline-aware multitrack assembly for MiniMax H3 Context Loop.

This node is additive: the original MiniMaxH3ChainAssemble contract remains
unchanged. It uses the existing assembler for video and source-track recovery,
then performs an absolute-frame audio mix as a final, separate mux.
"""
from __future__ import annotations

import os
import shutil
import uuid
from typing import Any

from . import chain_nodes as core


def _build_scene_envelopes(
    total_samples: int,
    sample_rate: int,
    scene_ranges: list[tuple[int, int]],
    source_gain: float,
    generated_gain: float,
    duck_gain: float,
    fade_in_seconds: float,
    fade_out_seconds: float,
):
    """Create independent source-duck and generated-overlay envelopes.

    Scene ranges are absolute (start_sample, end_sample) pairs on the final
    delivered timeline. Envelopes are sample-domain, so frame/sample rounding
    happens once at each absolute frame boundary.
    """
    np = core.np
    if np is None:
        raise RuntimeError("Advanced audio assembly requires NumPy.")
    total_samples = int(total_samples)
    if total_samples < 0 or int(sample_rate) <= 0:
        raise ValueError("Audio envelope dimensions must be non-negative.")
    values = (source_gain, generated_gain, duck_gain,
              fade_in_seconds, fade_out_seconds)
    if any(not core.math.isfinite(float(value)) for value in values):
        raise ValueError("Audio gains and fade durations must be finite.")
    if source_gain < 0 or generated_gain < 0:
        raise ValueError("Source and generated gains cannot be negative.")
    if not 0.0 <= duck_gain <= 1.0:
        raise ValueError("Ducking gain must be between 0 and 1.")
    if fade_in_seconds < 0 or fade_out_seconds < 0:
        raise ValueError("Fade durations cannot be negative.")

    source = np.full(total_samples, float(source_gain), dtype=np.float32)
    generated = np.zeros(total_samples, dtype=np.float32)
    fade_in = int(round(float(fade_in_seconds) * sample_rate))
    fade_out = int(round(float(fade_out_seconds) * sample_rate))
    for raw_start, raw_end in scene_ranges:
        start = max(0, min(total_samples, int(raw_start)))
        end = max(start, min(total_samples, int(raw_end)))
        length = end - start
        if not length:
            continue
        in_count = min(fade_in, length)
        out_count = min(fade_out, length)
        envelope = np.ones(length, dtype=np.float32)
        if in_count:
            envelope[:in_count] = np.linspace(
                0.0, 1.0, in_count, endpoint=True, dtype=np.float32)
        if out_count:
            envelope[length - out_count:] = np.minimum(
                envelope[length - out_count:],
                np.linspace(1.0, 0.0, out_count, endpoint=True,
                            dtype=np.float32))
        # Generated audio fades in/out while source audio ducks smoothly.
        generated[start:end] = np.maximum(
            generated[start:end], envelope * float(generated_gain))
        source[start:end] = np.minimum(
            source[start:end],
            float(source_gain) *
            (1.0 - (1.0 - float(duck_gain)) * envelope))
    return source, generated


class MiniMaxH3ChainAssembleAdvance(core.MiniMaxH3ChainAssemble):
    """Source-song bed plus scene-local H3 audio with gain and ducking."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"] = dict(inputs["required"])
        inputs["required"]["audio_source"] = (["source", "none"], {
            "default": "source",
            "tooltip": "Base track for the multitrack mix. Source preserves "
                       "the recovered original timeline; none starts silent. "
                       "Generated H3 scene audio is overlaid separately."})
        inputs["required"].update({
            "source_gain": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                "tooltip": "Linear gain for the continuous source/song bed."}),
            "generated_gain": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                "tooltip": "Linear gain for eligible independent H3 scene audio."}),
            "duck_gain": ("FLOAT", {
                "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.02,
                "tooltip": "Source-bed gain while an H3 overlay is active."}),
            "fade_in_seconds": ("FLOAT", {
                "default": 0.6, "min": 0.0, "max": 10.0, "step": 0.1,
                "tooltip": "Overlay fade-in and source duck attack."}),
            "fade_out_seconds": ("FLOAT", {
                "default": 0.7, "min": 0.0, "max": 10.0, "step": 0.1,
                "tooltip": "Overlay fade-out and source recovery."}),
            "overlay_scenes": ("STRING", {
                "default": "auto",
                "tooltip": "auto overlays scenes whose effective source-audio "
                           "target is off and which have no scene-local "
                           "lip-sync asset. Use all, none, or comma-separated "
                           "scene indexes (for example 2,4) to override."}),
        })
        return inputs

    RETURN_NAMES = ("video_path",)
    DESCRIPTION = (
        "Advanced multitrack assembly. Keeps the source timeline as a "
        "continuous bed, overlays eligible independent H3 scene audio, "
        "ducks/fades the source around overlays, and saves a separate "
        ".h3.generated.wav before mixing. The original Assemble node is "
        "not modified.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    @staticmethod
    def _output_result(manifest: dict[str, Any], final_path: str,
                       status: str, output_copy: str | None = None):
        published = output_copy or final_path
        core._publish_final_review_preview(manifest, published, status)
        return {
            "ui": {
                "text": [status],
                "images": [core._video_output_item(published)],
                "animated": (True,),
            },
            "result": (final_path,),
        }

    @staticmethod
    def _load_pcm(path: str, label: str):
        audio_source = core._make_source_timeline(audio_path=path)
        audio = core._source_timeline_source_audio(audio_source)
        if audio is None:
            return None
        waveform, rate = core._audio_waveform_3d(audio, label)
        waveform = waveform.detach().to(
            device="cpu", dtype=core.torch.float32).contiguous()
        return {"waveform": waveform, "sample_rate": int(rate)}

    @staticmethod
    def _scene_selection(value: str, manifest: dict[str, Any],
                         editorial_records: list[dict[str, Any]]):
        value = str(value or "auto").strip().lower()
        segments = {int(item["index"]): item
                    for item in manifest.get("segments", [])}
        if value == "none":
            return set()
        if value == "all":
            return {int(item["scene"]) for item in editorial_records
                    if item.get("kind") == "scene"}
        if value != "auto":
            selected = set()
            for token in value.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    scene = int(token)
                except ValueError as exc:
                    raise ValueError(
                        "overlay_scenes must be auto, all, none, or "
                        "comma-separated positive scene indexes.") from exc
                if scene < 1:
                    raise ValueError("Scene indexes in overlay_scenes start at 1.")
                selected.add(scene)
            known = set(segments)
            unknown = selected - known
            if unknown:
                raise ValueError("Unknown overlay scene indexes: %s" %
                                 ", ".join(str(item) for item in sorted(unknown)))
            return selected

        eligible = set()
        for index, segment in segments.items():
            # An explicit scene-local locked source target wins over the
            # Plan-wide policy. Existing lip-sync source clips are excluded
            # to prevent a duplicate source/dialogue overlay.
            policy = core._resolved_scene_audio_policy(manifest, segment)
            if (policy.get("source_audio_target", "off") != "locked"
                    and not segment.get("lip_sync_source_asset")):
                eligible.add(index)
        return eligible

    def assemble(
        self, manifest, audio_source, filename, audio_bitrate,
        *, overwrite_existing=False, copy_to_output=False,
        output_subfolder="", source_timeline=None, blend_schedule="plan",
        blend_video_vae=None, boundary_tone_match="off",
        color_stabilization="off", delete_checkpoints_after_assembly=False,
        generated_audio_join="av_overlap", source_gain=1.0,
        generated_gain=1.0, duck_gain=0.28, fade_in_seconds=0.6,
        fade_out_seconds=0.7, overlay_scenes="auto",
    ):
        if audio_source not in ("source", "none"):
            raise ValueError("Advanced Assemble audio_source must be source or none.")
        # Let the canonical assembler own all video assembly, editorial edits,
        # subtitles, source recovery, and checkpoint-cleanup validation.
        base_result = super().assemble(
            manifest, audio_source, filename, audio_bitrate,
            overwrite_existing=overwrite_existing, copy_to_output=False,
            output_subfolder=output_subfolder,
            source_timeline=source_timeline, blend_schedule=blend_schedule,
            blend_video_vae=blend_video_vae,
            boundary_tone_match=boundary_tone_match,
            color_stabilization=color_stabilization,
            delete_checkpoints_after_assembly=delete_checkpoints_after_assembly,
            generated_audio_join=generated_audio_join,
        )
        final_path = str(base_result["result"][0])
        status = str((base_result.get("ui") or {}).get(
            "text", ["assembled video"])[0])
        stem, _extension = os.path.splitext(final_path)
        generated_path = stem + ".generated.wav"
        preserved_path = stem + ".h3.generated.wav"

        # The sidecar is copied byte-for-byte before any gain/duck/mix stage.
        if os.path.isfile(generated_path):
            if os.path.abspath(generated_path) != os.path.abspath(preserved_path):
                shutil.copy2(generated_path, preserved_path)

        # Resolve the same validated/editorial frame map used by the base node.
        working_manifest = manifest
        upscale_manifest = None
        if str((manifest or {}).get("format") or "").startswith(
                "h3_chain_upscale_"):
            from . import upscale_nodes as upscale_support
            upscale_manifest = core._json_document(manifest)
            upscale_segments = upscale_support._validate_upscale_manifest(
                upscale_manifest)
            working_manifest = upscale_support._assembly_manifest(
                upscale_manifest, upscale_segments)
        segments = core._validate_manifest(working_manifest)
        run_name = core._strict_run_name(working_manifest.get("run_name"))
        editorial = core._manifest_editorial(working_manifest)
        editorial, records, extension_frames = (
            core._editorial_timeline_records(run_name, segments, editorial))
        prelude = core._validate_prelude(working_manifest)
        prelude_frames = int(prelude["frame_count"]) if prelude else 0
        total_frames = prelude_frames + int(extension_frames)
        selected_scenes = self._scene_selection(
            overlay_scenes, working_manifest, records)

        if not os.path.isfile(generated_path):
            # Older manifests do not always publish the sidecar by default.
            # Recover it with the same canonical scene/editorial/prelude
            # helpers, then persist the untouched assembled H3 track.
            try:
                recovered = core._generated_audio(
                    working_manifest, audio_join_mode=generated_audio_join)
                recovered = core._audio_with_scene_lip_sync(
                    recovered, working_manifest, "generated")
                recovered = core._audio_with_editorial_timeline(
                    recovered, records,
                    int(working_manifest["total_delivered_frames"]),
                    int(extension_frames), "Advanced generated editorial audio")
                if prelude is not None:
                    recovered = core._audio_with_prelude(
                        recovered, int(extension_frames), prelude)
                core._atomic_wav(recovered, generated_path)
                shutil.copy2(generated_path, preserved_path)
            except Exception as exc:
                raise RuntimeError(
                    "Advanced Assemble could not recover its independent H3 "
                    "audio sidecar: %s" % exc) from exc
        if not selected_scenes:
            output_copy = (core._copy_final_to_output(
                final_path, output_subfolder) if copy_to_output else None)
            if preserved_path and os.path.isfile(preserved_path):
                status += "; unmodified H3 audio sidecar -> " + preserved_path
            if output_copy:
                status += "; output copy -> " + output_copy
            return self._output_result(
                working_manifest, final_path, status, output_copy)

        generated = self._load_pcm(generated_path, "H3 generated sidecar")
        if generated is None:
            raise ValueError("Generated H3 sidecar contains no audio.")
        sample_rate = int(generated["sample_rate"])
        total_samples = core.sample_boundary_from_frames(
            total_frames, sample_rate, core.FPS)
        generated = core._resample_audio_exact(
            generated, sample_rate, total_samples,
            int(generated["waveform"].shape[1]),
            "Advanced H3 generated audio")

        source = self._load_pcm(final_path, "Advanced source-bed audio")
        if audio_source == "source" and source is None:
            raise ValueError(
                "Advanced Assemble selected source audio, but the assembled "
                "video has no audio stream. Connect a Source Timeline with "
                "the original song or select audio_source=none.")
        if source is None:
            channels = int(generated["waveform"].shape[1])
            source = {"waveform": core.torch.zeros(
                (1, channels, total_samples), dtype=core.torch.float32),
                "sample_rate": sample_rate}
        source_rate = int(source["sample_rate"])
        source = core._resample_audio_exact(
            source, sample_rate, total_samples,
            int(generated["waveform"].shape[1]),
            "Advanced source audio")

        ranges = []
        for record in records:
            if record.get("kind") != "scene":
                continue
            scene = int(record["scene"])
            if scene not in selected_scenes:
                continue
            start_frame = prelude_frames + int(record["start_frame"])
            end_frame = start_frame + int(record["frame_count"])
            ranges.append((
                core.sample_boundary_from_frames(start_frame, sample_rate,
                                                core.FPS),
                core.sample_boundary_from_frames(end_frame, sample_rate,
                                                core.FPS),
            ))
        source_envelope, generated_envelope = _build_scene_envelopes(
            total_samples, sample_rate, ranges,
            float(source_gain), float(generated_gain), float(duck_gain),
            float(fade_in_seconds), float(fade_out_seconds))

        np = core.np
        source_pcm = source["waveform"].numpy()[0]
        generated_pcm = generated["waveform"].numpy()[0]
        mixed = (source_pcm * source_envelope[None, :]
                 + generated_pcm * generated_envelope[None, :])
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.98:
            # Apply a transparent peak limiter (single scalar, no clipping)
            # rather than distorting individual samples or altering sidecar.
            mixed *= (0.98 / peak)
            core._LOG.warning(
                "Advanced H3 audio mix peak %.3f; applied %.4f peak safety gain.",
                peak, 0.98 / peak)
        mixed_audio = {
            "waveform": core.torch.from_numpy(
                np.ascontiguousarray(mixed[None, ...], dtype=np.float32)),
            "sample_rate": sample_rate,
        }
        mixed_tmp = stem + ".advanced-mix.tmp.wav"
        final_tmp = stem + ".advanced-mux.tmp.mp4"
        try:
            core._atomic_wav(mixed_audio, mixed_tmp)
            ffmpeg = core._usable_ffmpeg()
            if ffmpeg:
                core._run_ffmpeg([
                    ffmpeg, "-y", "-i", final_path, "-i", mixed_tmp,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "%dk" % int(audio_bitrate),
                    "-t", "%.9f" % (total_frames / float(core.FPS)),
                    "-map_metadata", "0",
                    "-movflags", "use_metadata_tags+faststart", final_tmp,
                ])
            elif core.av is not None:
                core._pyav_mux_audio(
                    final_path, mixed_audio, final_tmp,
                    int(audio_bitrate), total_frames)
            else:
                raise RuntimeError(
                    "Advanced multitrack mux requires ffmpeg or PyAV.")
            os.replace(final_tmp, final_path)
        finally:
            core._safe_unlink(mixed_tmp)
            core._safe_unlink(final_tmp)

        output_copy = (core._copy_final_to_output(
            final_path, output_subfolder) if copy_to_output else None)
        status += (
            "; advanced multitrack: source gain %.2f, H3 gain %.2f, duck %.2f; "
            "overlaid scenes [%s]; fades %.2fs/%.2fs; exact %d-frame mux"
            % (float(source_gain), float(generated_gain), float(duck_gain),
               ",".join(str(item) for item in sorted(selected_scenes)),
               float(fade_in_seconds), float(fade_out_seconds), total_frames))
        status += "; unmodified H3 audio sidecar -> " + preserved_path
        if output_copy:
            status += "; output copy -> " + output_copy
        return self._output_result(
            working_manifest, final_path, status, output_copy)
