# Source

This package is the Who&When Pro evaluation harness, copied unmodified from the
authors' repository (https://github.com/whowhenpro/whowhen_pro, commit 14369dc). It is
used here for stage 5 (all-at-once failure attribution and scoring).

Run from the repository root, e.g.:

    python -m whowhen_eval.run --model claude-sonnet-4-6 --data-root outputs/eval_data \
        --modality text --out outputs/eval_results

Its dependencies (litellm, pyyaml, pillow) are listed in the root `requirements.txt`.

## License

The original package declares the MIT license (`license = { text = "MIT" }` in its
`pyproject.toml`); the upstream repository did not include a separate LICENSE file.

MIT License

Copyright (c) 2026 The Who&When Pro authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
