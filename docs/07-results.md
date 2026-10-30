# Chapter 7 · Results: what did it learn?

Numbers are nice, but the real test is reading what it writes.

```bash
python sample.py --prompt "Once upon a time, there was a unicorn named" --n 3
```

## After 6 minutes (step 1,000, val loss 1.81)

> Once upon a time, there was a big, bright sun in the sky. The sun was shining, and the flowers were singing. The sun would fly and the clouds would fly. The sun was very happy.
> One day, a little girl named Lily went for a walk. She saw a big tree with a hole in it. She was very curious about this. She walked closer to the hole and started to dig. She dug and dug until she found a small hole. She was happy!
> Lily worked hard to make the hole in. She dug and dug until she found a tiny, shiny rock. She showed the rock to her mom. "Look, Mom! I found a rock!" Lily said. Her mom smiled, and they both laughed.

Already it has **grammar, a named character, dialogue in quotes, and a simple plot**. But the world is wobbly: "the flowers were singing", "the sun would fly", and "worked hard to make the hole in".

## After 70 minutes (step 12,000, val loss 1.19)

> Once upon a time, there was a unicorn named Poppy. Poppy was a happy and healthy unicorn. She lived in a big forest with her friends. Every day, Poppy would play with her friends and have fun.
> One day, Poppy met a small mouse. The mouse said, "Hi, Poppy! I am hungry. Can we share?" Poppy thought for a moment and said, "Yes, we can share. We can eat and play together. We are all healthy and strong."
> So, Poppy and the mouse became good friends. They played and ate together every day. They were very happy and healthy in the big forest. And they always remembered the day Poppy shared her food with her new friend, the little mouse.

A complete story: **setup → event → dialogue → resolution → a small moral**, with consistent characters and a callback at the end ("the day Poppy shared her food").

## Where it still slips

Same prompt, different random sample:

> Once upon a time, there was a unicorn named **Booy**. Booy was a very silly **penguin** who loved to play and laugh all day. He lived in a big, icy pond with his friend, a little seal named Lulu. […] **Lily** said, "Boo, let's climb the tree…"

- **The unicorn turns into a penguin.** Our prompt ended mid-sentence, the model picked an odd name ("Booy" = `␣Bo` + `oy`), and from then on "penguin" was simply more probable. Small models follow *local* probability more than the *global* plan.
- **"Lulu" briefly becomes "Lily".** "Lily" is one of the most common names in TinyStories, so it pulls hard.

> [!TIP]
> **🧠 Why this matters:** these failures are exactly why chat models are trained in a second stage to *follow instructions*. [Chapter 9](09-whats-next.md) plans that next step for our model.

## Temperature: creative vs safe

`sample.py --temperature` changes how adventurous the sampling is. The model's scores are divided by the temperature before turning them into probabilities:

| Temperature | What our model actually did with "…a unicorn named" |
|---|---|
| 0.3 | Clean but predictable: the unicorn is named **Lily** (the most common name in the data), meets a monster, scares it off with her horn, "lived happily ever after" |
| **0.8** (default) | A good balance: Poppy and the mouse who learn to share |
| 1.3 | More surprising word choices ("a *flexible* unicorn with a shiny crown", "He would *suffer* a lot", "a fast *reindeer*"), and the plot starts to wander |

We also use **top-k = 50**: only the 50 most likely tokens are ever considered, which cuts off the long tail of nonsense.

## How does it compare?

| | Our model | Qwen3 14B (via Ollama) |
|---|---|---|
| Parameters | 50.9M | 14,800M (~290× more) |
| Training data | 393M tokens of children's stories | trillions of tokens of everything |
| Training time | 70 minutes on one consumer GPU | months on thousands of datacenter GPUs |
| Writes a coherent children's story | ✅ usually | ✅ always |
| Follows "a unicorn with a happy ending" | ❌ not yet | ✅ |
| Speed on the RX 7900 XT | 120–310 tokens/sec | 63 tokens/sec |

That last row is the interesting one: small, specialized models are *fast*. Pairing a big model's understanding with a small model's speed is exactly what Story Studio does next.

---
← [Chapter 6 · Training](06-training.md) · Next: [Chapter 8 · Story Studio →](08-story-studio.md)
