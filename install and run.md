# Clone the repo
```bash
git clone https://github.com/0xShug0/audio.cpp.git
cd audio.cpp
```
# Build audio.cpp
```bash
# For GPU
scripts/build_linux.sh --backend cuda --cuda-arch native --target audiocpp_cli --target audiocpp_server

# For CPU
scripts/build_linux.sh --backend cpu --target audiocpp_cli --target audiocpp_server
```

# Check available models
```bash
python3 tools/model_manager_v2.py list
```

# Install ASR models
```bash
python3 tools/model_manager_v2.py install nemotron_asr_f16
```

## Test (Use short audio clip)
```bash
ffmpeg -y -i assets/resources/speech.wav -ar 16000 -ac 1 assets/resources/speech_16k.wav # For long audio

# For GPU
build/linux-cuda-release/bin/audiocpp_cli --task asr --family nemotron_asr \
  --model models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  --backend cuda --mode streaming --audio assets/resources/speech_16k.wav --language en-US \
  --text-out transcript.txt

# For CPU
build/linux-cpu-release/bin/audiocpp_cli --task asr --family nemotron_asr \
  --model models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  --backend cpu --mode streaming --audio assets/resources/speech_16k.wav --language en-US \
  --text-out transcript.txt

```

## Measure inference time

`--metrics` doesn't support streaming mode, so for the full long audio file
(run with `--mode streaming`), use `--log` instead and read `session.wall_ms`

```bash
build/linux-cuda-release/bin/audiocpp_cli --task asr --family nemotron_asr \
  --model models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  --backend cuda --mode streaming --audio assets/resources/speech_16k.wav --language en-US \
  --request-option lookahead_tokens=13 --log --text-out transcript.txt 2>&1 | grep session.wall_ms

```
## ASR time comparison
input: 6 min 58 sec
audio.cpp (cuda) infer time: 4.33 sec
python nemo file (cuda) infer time: 4.49 sec


# Install a lightweight TTS model (Kokoro 82M)
```bash
python3 tools/model_manager_v2.py install kokoro_82m_q8_0
```

## Test TTS
Kokoro's English/Spanish/French/Hindi/Italian/Portuguese voices need eSpeak-ng
installed (`sudo apt install espeak-ng`, or build with
`-DAUDIOCPP_STATIC_ESPEAK=ON`, see `docs/espeak_phonemizer.md`). Chinese and
Japanese use bundled G2P tables and need no extra setup, so that's the
quickest path to a first test:

```bash
build/linux-cuda-release/bin/audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --backend cpu \
  --language zh \
  --voice-id zf_xiaobei \
  --text "你好，这是中文语音测试。" \
  --out out_zh.wav
```

With eSpeak-ng installed, English works the same way:

```bash
build/linux-cuda-release/bin/audiocpp_cli --task tts --family kokoro_tts \
  --model models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf \
  --backend cuda \
  --language en-us \
  --voice-id af_heart \
  --metrics \
  --out out.wav \
  --text "Friends, neighbors, and fellow builders of the common life we meet at a time when it is easy to confuse noise with strength and speed with progress.  The machines around us move quickly the markets change quickly.  The messages arrive without rest, and yet the great work of a people is still done by patience courage, memory and care a nation is not made by its engines alone, nor by the number of wires that cross its cities nor by the height of its towers.  It is made by the character of ordinary people doing ordinary duties with unusual faithfulness.  There is a kind of courage that is praised in every age the courage of the field, the courage of the rescue, the courage that runs toward visible danger we honor it and rightly but there is another courage quieter and more frequently demanded.  It is the courage to tell the truth when exaggeration is profitable to be fair when anger is fashionable to study a question when slogans would be easier to keep a promise when no audience is watching and to labor for a future whose benefits may be enjoyed by people we will never meet the life of a free community asks more of us than complaint.  Complaint may be the beginning of wisdom when it names a real injury, but it becomes a poor substitute for service when it never lifts a hand.  The citizen who only sneers at every failure is not yet doing the full work of citizenship.  The citizen who sees a broken school, a neglected park, a lonely elder, a dishonest custom, or a bitter quarrel, and asks, what can I repair?  What can I learn?  Whom can I help has already begun to strengthen the republic?  We should not pretend that the tasks before us are small.  Many families know the weight of prices that rise faster than wages many young people wonder whether the promises offered to them were written for another generation.  Many workers feel that their skill has been treated as disposable.  Many communities have watched industries vanish rivers become polluted neighbors become strangers, and public speech become a contest of contempt.  To name these things is not pessimism it is honesty, but honesty must not end in despair.  A problem clearly seen is a summons, not a sentence.  The remedy begins with a principal so simple that sophisticated people often overlook it every human being has a dignity that does not depend on wealth, fashion, party, accent, origin, age, or usefulness to the powerful from that principle flow many duties we must build schools that do more than sort children into winners and losers.  We must build workplaces where excellence is expected and exploitation is refused.  We must build towns where safety does not require suspicion of every stranger, and where prosperity does not mean pushing the poor out of sight.  We must build institutions that can admit error without collapsing into shame and correct error without surrendering to cynicism no generation receives a finished country each receives an inheritance mix with achievement and failure.  We inherit bridges, laws, libraries, songs, farms, courts, workshops, and the accumulated skill of millions.  We also inherit debts, exclusions, wounds, and habits of mind that no longer deserve obedience.  Gratitude does not require blindness, and criticism does not require contempt.  The mature patriot loves a country as one loves a family, not because it is flawless, but because it is bound to us by duty, memory, sacrifice, and hope.  If we want a public life worthy of free people, we must recover the art of listening without surrendering judgment.  Listening is not weakness it is the discipline by which a large society remains human.  When we listen well, we do not merely wait for our turn to speak.  We search for the fear beneath the anger, the experience beneath the opinion, the fact beneath the rumor, and the possible friend beneath the apparent opponent.  The strongest argument is not weakened by understanding the person who disagrees with it it is strengthened because it has passed through the test of reality.  We must also recover respect for craft.  There is craft in teaching a child to read in repairing a roof, in writing clear code, in planting an orchard, in nursing the sick, and keeping books honestly, in designing a bridge, in cooking a meal, in conducting research, in raising a family, and in governing a city, a society that honors only fame and fortune teaches its people to despise the very work that keeps it alive.  Let us praise excellence wherever it appears, especially when it appears without applause.  The young should hear from us neither flattery nor scolding, but invitation they should be told that their lives matter too much to be spent only in performance and comparison.  They should be given the tools to think the room to fail the example of adults who can apologize, and the challenge of work that is larger than self display.  If they inherit only our anxieties, they will grow tired before their strength has flowered.  If they inherit our courage, our unfinished questions, and our confidence that repair is possible, they may exceed us in ways we cannot imagine.  Technology will help us, but it will not absolve us.  A tool can multiply wisdom or multiply folly.  It can carry knowledge across oceans or carry falsehood faster than correction can follow.  It can free hands from drudgery or make every waking hour answerable to a silent machine.  The question is not whether we will have powerful tools we already do.  The question is whether we will have powerful purposes disciplined enough to guide them toward human flourishing rather than mere appetite.  Let us therefore choose a standard higher than victory over one another.  Let us choose usefulness.  Let the engineer ask whether the design serves life.  Let the official ask whether the rule is just.  Let the merchant ask whether profit has been earned without degrading the buyer or the worker.  Let the artist ask whether beauty has been made more available.  Let the neighbor ask who has been missing from the table.  Let each of us ask at the close of the day whether some small corner of the world is more honest, more kind, or more capable because we pass through it there will be disappointments.  Good laws will be delayed good plans will meet resistance.  Good people will misunderstand one another.  Some efforts will fail after honest labor, but the measure of a people is not that they never stumble.  It is that they know how to begin again without surrendering their souls to bitterness.  The builders of every decent institution worked in the presence of uncertainty they planted trees without proof that they would sit in the shade they taught children whose futures they could not control.  They defended principles before those principles were popular.  They endured being called foolish by those who mistook caution for wisdom.  So let us be practical in the deepest sense.  Let us repair what is near, learn what is true, protect what is vulnerable and imagine what is better.  Let us argue with seriousness and serve with humility.  Let us refuse the cheap comfort of believing that every fault belongs to someone else.  Let us remember that a republic is not a spectator event.  It is a shared workshop, a common table, a school of character, and a promise renewed by use.  The work is large, but we are not helpless.  The future is uncertain, but it is not empty.  It waits for the shape of our hands, our words, our laws, our mercy, and our courage.  If we do these things not perfectly but faithfully, we may leave to those who follow us something better than an inheritance of complaint.  We may leave them proof that free people can govern themselves, correct themselves, and care for one another without losing the fire of liberty.  We may leave them towns where trust has been rebuilt institutions that deserve patience work that carries dignity and stories that teach courage and when they ask what we did in our season of difficulty, may the answer be simple.  We did not look away we did not give up, and we gave our strength to the common good
"
```
Infer time: 14.4 sec

# Run as a server (REST API)

`audiocpp_server` serves models over an OpenAI-style HTTP API, so any Python
project can call it by model name over the network instead of shelling out to
`audiocpp_cli`. It needs a config file declaring which models are servable
under which id.

`server.json` (already created at the repo root; paths are relative to this
file's own location, not your shell's cwd):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "lazy_load": true,
  "models": [
    {
      "id": "kokoro",
      "family": "kokoro_tts",
      "path": "models/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
      "task": "tts",
      "mode": "offline"
    },
    {
      "id": "nemotron",
      "family": "nemotron_asr",
      "path": "models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf",
      "task": "asr",
      "mode": "offline"
    }
  ]
}
```

Start the server:

```bash
build/linux-cuda-release/bin/audiocpp_server --config server.json
```

With `"lazy_load": true`, each model loads into memory on its first request,
not at startup.

## Call it with curl

```bash
# TTS
curl -s http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"你好，这是中文语音测试。","voice":"zf_xiaobei","language":"zh"}' \
  -o out.wav

# ASR
curl -s http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@assets/resources/sample_16k.wav" \
  -F "model=nemotron" \
  -F "language=en-US"
```

## Call it from Python

```python
import requests

BASE = "http://127.0.0.1:8080"

# TTS: POST /v1/audio/speech
r = requests.post(f"{BASE}/v1/audio/speech", json={
    "model": "kokoro",
    "input": "你好，这是中文语音测试。",
    "voice": "zf_xiaobei",
    "language": "zh",
})
open("out.wav", "wb").write(r.content)

# ASR: POST /v1/audio/transcriptions (multipart file upload)
with open("speech.wav", "rb") as f:
    r = requests.post(f"{BASE}/v1/audio/transcriptions",
                       files={"file": f},
                       data={"model": "nemotron", "language": "en-US"})
print(r.json())
# {'text': '...', 'timing': {'wall_ms':64.9,'audio_duration_ms':14071.9,'rtf':0.0046}}
```

The ASR response includes `timing` (wall_ms, audio_duration_ms, rtf) for free
— no separate `--metrics`/`--log` step needed when going through the server.

Other endpoints (`audiocpp_server --help` lists all of them): `GET /v1/models`
(list configured/loaded models), `POST /v1/audio/speech/live` and
`/v1/audio/transcriptions/live` (SSE streaming for real-time use),
`/v1/batches/transcriptions` (batch multi-file), `/v1/audio/alignments`.
There's also an embedded WebUI (`audiocpp_server --ui --ui-management`) if you
want a browser GUI instead of/alongside the API.
