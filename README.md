# PreservedNV-STS2

PreservedNV-STS2 is an inference-only patch and example suite for preserving
multiple non-verbal (NV) regions from reference audio during OmniVoice voice
cloning.

```text
reference audio → optional PretrainedSED intervals → fixed NV tokens → generated WAV
```

The repository contains no model weights, audio data, dataset copy, generated
audio, private manifest, or OmniVoice fork. It depends only on public upstream
projects installed outside this directory.

## Patch scope

`patches/0001-multi-interval-nv-fixing.patch` modifies the official OmniVoice
inference implementation at `omnivoice/models/omnivoice.py`. It adds
NV-conditioned voice cloning with either one `nv_position=(start, end)` or
multiple `nv_position=[(start, end), ...]` intervals.

The patch validates, sorts, and merges touching intervals; encodes each
reference NV span; inserts fixed spans into the diffusion target; masks those
spans; and removes them from the iterative decoding schedule.

The patch is tested against official OmniVoice commit
`9e21256d6a32e3e82177d84063fb1c319d651ce9`. This repository does not require
or reference a private source repository. For ordinary use, `git apply --check`
is the compatibility check. The pinned commit is only needed to reproduce the
exact tested software revision.

## Install

```bash
git clone https://github.com/k2-fsa/OmniVoice.git OmniVoice
cd OmniVoice
git apply --check /path/to/PreservedNV-STS2/patches/0001-multi-interval-nv-fixing.patch
git apply /path/to/PreservedNV-STS2/patches/0001-multi-interval-nv-fixing.patch
pip install -e .
```

For the exact tested base, insert this optional command before applying:

```bash
git checkout 9e21256d6a32e3e82177d84063fb1c319d651ce9
```

Install this repository's helper package:

```bash
cd /path/to/PreservedNV-STS2
pip install -e '.[dev,example]'
pytest
```

To use SED, separately install the public
[PretrainedSED](https://github.com/fschmid56/PretrainedSED) project and its
requirements. Its BEATs strong-event checkpoint is fetched/managed by that
project and is not redistributed here.

```bash
git clone https://github.com/fschmid56/PretrainedSED.git /path/to/PretrainedSED
cd /path/to/PretrainedSED
pip install -r requirements.txt
```

## SED-to-NV-fixing command

```bash
python examples/detect_and_fix.py \
  --audio ./reference.wav \
  --reference-text 'A reference transcript.' \
  --text 'Text to synthesize.' \
  --language English \
  --model /path/to/OmniVoice-model \
  --pretrained-sed-root /path/to/PretrainedSED \
  --device cuda:0 \
  --threshold 0.20 \
  --output outputs/nv_fixed.wav
```

`--dry-run` performs SED and writes the JSON audit only. If no event is found,
the default generates ordinary voice cloning; use `--no-event error` to stop.
Use `--event-label` repeatedly to replace the default human non-verbal event
set. The audit records selected labels, scores, and intervals.

## NonVerbalSpeech38k Hugging Face example

`examples/nv38k_hf_example.py` downloads one row from
[`nonverbalspeech/nonverbalspeech38k`](https://huggingface.co/datasets/nonverbalspeech/nonverbalspeech38k)
and mirrors the single-row inference pattern in OmniVoice's `infer.ipynb`.
The default is `train[611]`, with the Korean target sentence used in that
notebook. Use `--index`, `--text`, and `--language` to select another example.

For a ground-truth-boundary smoke test:

```bash
python examples/nv38k_hf_example.py \
  --model /path/to/OmniVoice-model \
  --device cuda:0 \
  --index 611 \
  --interval-source dataset \
  --output outputs/nv38k_611_dataset.wav
```

For the end-to-end SED pipeline on the same Hugging Face waveform:

```bash
python examples/nv38k_hf_example.py \
  --model /path/to/OmniVoice-model \
  --pretrained-sed-root /path/to/PretrainedSED \
  --device cuda:0 \
  --index 611 \
  --interval-source sed \
  --threshold 0.20 \
  --output outputs/nv38k_611_sed.wav
```

Use `--dry-run` to download a row and write an audit without loading
OmniVoice. If Hugging Face requires authentication or applies rate limits, run
`huggingface-cli login` first. Dataset audio is downloaded at runtime and is
ignored by Git.

## Direct model API

```python
from omnivoice.models.omnivoice import OmniVoice

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")
prompt = model.create_nv_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="A reference transcript.",
    nv_position=[(0.72, 1.12), (2.30, 2.65)],
)
audio = model.generate(
    text="Text to synthesize.",
    language="English",
    nv_voice_clone_prompt=prompt,
)
```

Intervals are seconds in the reference waveform. Use an output duration aligned
to the reference when temporal position matters; a materially different target
duration can shift the perceived NV location.

## License and attribution

The original helper code and independently authored modifications are licensed under the MIT License. The OmniVoice patch is derived from Apache-2.0-licensed upstream code and retains the applicable Apache-2.0 license requirements, including copyright notices and attribution in the NOTICE file. PretrainedSED is an optional MIT-licensed dependency. Model and dataset terms remain the responsibility of their respective upstream providers.
