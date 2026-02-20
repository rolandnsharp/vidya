/**
 * The most atomic way to train and run inference for a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * Ported from @karpathy's microgpt.py
 */

const fs = require('fs');
const https = require('https');

// --- Seeded PRNG (Mulberry32) to replace random.seed(42) ---
let _seed = 42;
function mulberry32() {
  _seed |= 0; _seed = _seed + 0x6D2B79F5 | 0;
  let t = Math.imul(_seed ^ _seed >>> 15, 1 | _seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}
function randGauss(mean = 0, std = 1) {
  let u, v, s;
  do { u = 2 * mulberry32() - 1; v = 2 * mulberry32() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  return mean + std * u * Math.sqrt(-2 * Math.log(s) / s);
}
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(mulberry32() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
function weightedChoice(weights) {
  let total = 0;
  for (const w of weights) total += w;
  let r = mulberry32() * total;
  for (let i = 0; i < weights.length; i++) { r -= weights[i]; if (r <= 0) return i; }
  return weights.length - 1;
}

// --- Download input.txt if needed, then run ---
function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, res => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        file.close();
        fs.unlinkSync(dest);
        return downloadFile(res.headers.location, dest).then(resolve, reject);
      }
      res.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', err => { fs.unlinkSync(dest); reject(err); });
  });
}

// --- CLI argument parsing ---
const args = process.argv.slice(2);
let promptText = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--prompt' && i + 1 < args.length) promptText = args[i + 1];
}
const WEIGHTS_FILE = 'weights.json';

async function main() {
  // Let there be a Dataset `docs`: list of documents (e.g. a list of names)
  if (!fs.existsSync('input.txt')) {
    const url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt';
    await downloadFile(url, 'input.txt');
  }
  let docs = fs.readFileSync('input.txt', 'utf-8').split('\n').map(l => l.trim()).filter(Boolean);
  shuffle(docs);
  console.log(`num docs: ${docs.length}`);

  // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
  const uchars = [...new Set(docs.join(''))].sort();
  const BOS = uchars.length;
  const vocab_size = uchars.length + 1;
  console.log(`vocab size: ${vocab_size}`);

  // Let there be Autograd to recursively apply the chain rule through a computation graph
  class Value {
    constructor(data, children = [], localGrads = []) {
      this.data = data;
      this.grad = 0;
      this._children = children;
      this._localGrads = localGrads;
    }

    add(other) {
      if (!(other instanceof Value)) other = new Value(other);
      return new Value(this.data + other.data, [this, other], [1, 1]);
    }

    mul(other) {
      if (!(other instanceof Value)) other = new Value(other);
      return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    }

    pow(n) {
      return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
    }

    log() {
      return new Value(Math.log(this.data), [this], [1 / this.data]);
    }

    exp() {
      const e = Math.exp(this.data);
      return new Value(e, [this], [e]);
    }

    relu() {
      return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
    }

    neg() { return this.mul(-1); }
    sub(other) { return this.add(other instanceof Value ? other.neg() : new Value(other).neg()); }
    div(other) {
      if (!(other instanceof Value)) other = new Value(other);
      return this.mul(other.pow(-1));
    }

    backward() {
      const topo = [];
      const visited = new Set();
      function buildTopo(v) {
        if (!visited.has(v)) {
          visited.add(v);
          for (const child of v._children) buildTopo(child);
          topo.push(v);
        }
      }
      buildTopo(this);
      this.grad = 1;
      for (let i = topo.length - 1; i >= 0; i--) {
        const v = topo[i];
        for (let j = 0; j < v._children.length; j++) {
          v._children[j].grad += v._localGrads[j] * v.grad;
        }
      }
    }
  }

  // Helper: sum an array of Values
  function vsum(arr) {
    let s = arr[0];
    for (let i = 1; i < arr.length; i++) s = s.add(arr[i]);
    return s;
  }

  // Initialize the parameters, to store the knowledge of the model
  const n_layer = 1;
  const n_embd = 16;
  const block_size = 16;
  const n_head = 4;
  const head_dim = n_embd / n_head;

  function matrix(nout, nin, std = 0.08) {
    const m = [];
    for (let i = 0; i < nout; i++) {
      const row = [];
      for (let j = 0; j < nin; j++) row.push(new Value(randGauss(0, std)));
      m.push(row);
    }
    return m;
  }

  const state_dict = {
    wte: matrix(vocab_size, n_embd),
    wpe: matrix(block_size, n_embd),
    lm_head: matrix(vocab_size, n_embd),
  };
  for (let i = 0; i < n_layer; i++) {
    state_dict[`layer${i}.attn_wq`] = matrix(n_embd, n_embd);
    state_dict[`layer${i}.attn_wk`] = matrix(n_embd, n_embd);
    state_dict[`layer${i}.attn_wv`] = matrix(n_embd, n_embd);
    state_dict[`layer${i}.attn_wo`] = matrix(n_embd, n_embd);
    state_dict[`layer${i}.mlp_fc1`] = matrix(4 * n_embd, n_embd);
    state_dict[`layer${i}.mlp_fc2`] = matrix(n_embd, 4 * n_embd);
  }
  const params = [];
  for (const mat of Object.values(state_dict))
    for (const row of mat)
      for (const p of row)
        params.push(p);
  console.log(`num params: ${params.length}`);

  // Define the model architecture
  function linear(x, w) {
    return w.map(wo => vsum(wo.map((wi, i) => wi.mul(x[i]))));
  }

  function softmax(logits) {
    let maxVal = -Infinity;
    for (const v of logits) if (v.data > maxVal) maxVal = v.data;
    const exps = logits.map(v => v.sub(maxVal).exp());
    const total = vsum(exps);
    return exps.map(e => e.div(total));
  }

  function rmsnorm(x) {
    const ms = vsum(x.map(xi => xi.mul(xi))).div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map(xi => xi.mul(scale));
  }

  function gpt(token_id, pos_id, keys, values) {
    const tok_emb = state_dict.wte[token_id];
    const pos_emb = state_dict.wpe[pos_id];
    let x = tok_emb.map((t, i) => t.add(pos_emb[i]));
    x = rmsnorm(x);

    for (let li = 0; li < n_layer; li++) {
      // 1) Multi-head Attention block
      let x_residual = x;
      x = rmsnorm(x);
      const q = linear(x, state_dict[`layer${li}.attn_wq`]);
      const k = linear(x, state_dict[`layer${li}.attn_wk`]);
      const v = linear(x, state_dict[`layer${li}.attn_wv`]);
      keys[li].push(k);
      values[li].push(v);
      const x_attn = [];
      for (let h = 0; h < n_head; h++) {
        const hs = h * head_dim;
        const q_h = q.slice(hs, hs + head_dim);
        const k_h = keys[li].map(ki => ki.slice(hs, hs + head_dim));
        const v_h = values[li].map(vi => vi.slice(hs, hs + head_dim));
        const attn_logits = k_h.map(kt =>
          vsum(q_h.map((qj, j) => qj.mul(kt[j]))).div(Math.sqrt(head_dim))
        );
        const attn_weights = softmax(attn_logits);
        for (let j = 0; j < head_dim; j++) {
          x_attn.push(vsum(attn_weights.map((w, t) => w.mul(v_h[t][j]))));
        }
      }
      x = linear(x_attn, state_dict[`layer${li}.attn_wo`]);
      x = x.map((a, i) => a.add(x_residual[i]));
      // 2) MLP block
      x_residual = x;
      x = rmsnorm(x);
      x = linear(x, state_dict[`layer${li}.mlp_fc1`]);
      x = x.map(xi => xi.relu());
      x = linear(x, state_dict[`layer${li}.mlp_fc2`]);
      x = x.map((a, i) => a.add(x_residual[i]));
    }

    return linear(x, state_dict.lm_head);
  }

  // --- Save / Load weights ---
  function saveWeights() {
    const data = params.map(p => p.data);
    fs.writeFileSync(WEIGHTS_FILE, JSON.stringify({ uchars, data }));
    console.log(`\nweights saved to ${WEIGHTS_FILE}`);
  }

  function loadWeights() {
    if (!fs.existsSync(WEIGHTS_FILE)) {
      console.error(`Error: ${WEIGHTS_FILE} not found. Run training first (no --prompt flag).`);
      process.exit(1);
    }
    const saved = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf-8'));
    for (let i = 0; i < params.length; i++) params[i].data = saved.data[i];
    console.log(`weights loaded from ${WEIGHTS_FILE}`);
  }

  if (promptText !== null) {
    // --- Inference-only mode with a prompt ---
    loadWeights();
    const temperature = 0.5;
    const promptTokens = promptText.split('').map(ch => {
      const idx = uchars.indexOf(ch);
      if (idx === -1) { console.warn(`warning: character '${ch}' not in vocabulary, skipping`); return null; }
      return idx;
    }).filter(x => x !== null);
    const tokens = [BOS, ...promptTokens];

    const keys = Array.from({ length: n_layer }, () => []);
    const vals = Array.from({ length: n_layer }, () => []);
    const generated = [];

    // Feed prompt tokens through the model (prefill)
    for (let pos_id = 0; pos_id < tokens.length && pos_id < block_size; pos_id++) {
      const logits = gpt(tokens[pos_id], pos_id, keys, vals);
      // On the last prompt token, start sampling
      if (pos_id === tokens.length - 1) {
        const probs = softmax(logits.map(l => l.div(temperature)));
        let token_id = weightedChoice(probs.map(p => p.data));
        if (token_id !== BOS) generated.push(uchars[token_id]);
        // Continue generating
        for (let gen_pos = tokens.length; gen_pos < block_size; gen_pos++) {
          if (token_id === BOS) break;
          const logits2 = gpt(token_id, gen_pos, keys, vals);
          const probs2 = softmax(logits2.map(l => l.div(temperature)));
          token_id = weightedChoice(probs2.map(p => p.data));
          if (token_id === BOS) break;
          generated.push(uchars[token_id]);
        }
      }
    }
    console.log(`prompt: "${promptText}"`);
    console.log(`output: ${promptText}${generated.join('')}`);

  } else {
    // --- Training mode ---
    const learning_rate = 0.01, beta1 = 0.85, beta2 = 0.99, eps_adam = 1e-8;
    const m = new Float64Array(params.length);
    const v = new Float64Array(params.length);

    const num_steps = 1000;
    for (let step = 0; step < num_steps; step++) {
      const doc = docs[step % docs.length];
      const tokens = [BOS, ...doc.split('').map(ch => uchars.indexOf(ch)), BOS];
      const n = Math.min(block_size, tokens.length - 1);

      const keys = Array.from({ length: n_layer }, () => []);
      const vals = Array.from({ length: n_layer }, () => []);
      const losses = [];
      for (let pos_id = 0; pos_id < n; pos_id++) {
        const token_id = tokens[pos_id];
        const target_id = tokens[pos_id + 1];
        const logits = gpt(token_id, pos_id, keys, vals);
        const probs = softmax(logits);
        losses.push(probs[target_id].log().neg());
      }
      const loss = vsum(losses).div(n);

      loss.backward();

      const lr_t = learning_rate * (1 - step / num_steps);
      for (let i = 0; i < params.length; i++) {
        const p = params[i];
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
        const m_hat = m[i] / (1 - beta1 ** (step + 1));
        const v_hat = v[i] / (1 - beta2 ** (step + 1));
        p.data -= lr_t * m_hat / (Math.sqrt(v_hat) + eps_adam);
        p.grad = 0;
      }

      process.stdout.write(`\rstep ${String(step + 1).padStart(4)} / ${String(num_steps).padStart(4)} | loss ${loss.data.toFixed(4)}`);
    }

    // Save weights after training
    saveWeights();

    // Default inference: generate samples
    const temperature = 0.5;
    console.log('--- inference (new, hallucinated text) ---');
    for (let sample_idx = 0; sample_idx < 20; sample_idx++) {
      const keys = Array.from({ length: n_layer }, () => []);
      const vals = Array.from({ length: n_layer }, () => []);
      let token_id = BOS;
      const sample = [];
      for (let pos_id = 0; pos_id < block_size; pos_id++) {
        const logits = gpt(token_id, pos_id, keys, vals);
        const probs = softmax(logits.map(l => l.div(temperature)));
        token_id = weightedChoice(probs.map(p => p.data));
        if (token_id === BOS) break;
        sample.push(uchars[token_id]);
      }
      console.log(`sample ${String(sample_idx + 1).padStart(2)}: ${sample.join('')}`);
    }
  }
}

main();
