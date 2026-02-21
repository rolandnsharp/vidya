# Reinforcement Learning: From Sutton's Foundations to Vidya

A research paper and reference collecting key ideas, code, and connections from
[Rich Sutton's incompleteideas.net](http://incompleteideas.net/) and how they
might apply to our neurosymbolic AI model, **Vidya**.

---

## Table of Contents

1. [What is Reinforcement Learning?](#what-is-reinforcement-learning)
2. [The Bitter Lesson](#the-bitter-lesson)
3. [Verification: The Key to AI](#verification-the-key-to-ai)
4. [Core RL Algorithms in Lisp](#core-rl-algorithms-in-lisp)
   - [Temporal-Difference Learning (TD)](#temporal-difference-learning)
   - [Value Iteration & Dynamic Programming](#value-iteration--dynamic-programming)
   - [Monte Carlo Methods](#monte-carlo-methods)
   - [Q-Learning & Double Q-Learning](#q-learning--double-q-learning)
   - [Multi-Armed Bandits](#multi-armed-bandits)
   - [Policy Gradient / Gradient Bandits](#gradient-bandits)
   - [R-Learning (Average Reward)](#r-learning)
   - [Dyna Architecture (Planning + Learning)](#dyna-architecture)
   - [TD Model of Classical Conditioning](#td-model-of-classical-conditioning)
   - [Mountain Car with Tile Coding](#mountain-car-with-tile-coding)
   - [Tic-Tac-Toe (Value-Based Self-Play)](#tic-tac-toe)
5. [Lisp as an AI Substrate](#lisp-as-an-ai-substrate)
6. [Forth vs Lisp: Why Forth Wins for Vidya](#forth-vs-lisp-why-forth-wins-for-vidya)
7. [Applying RL to Vidya](#applying-rl-to-vidya)
8. [Key References & Links](#key-references--links)

---

## What is Reinforcement Learning?

From the [RL FAQ](http://incompleteideas.net/RL-FAQ.html) (Rich Sutton, 2004):

> Reinforcement learning (RL) is learning from interaction with an environment,
> from the consequences of action, rather than from explicit teaching.

The mathematical framework is the **Markov Decision Process (MDP)**:
- An **agent** interacts with an **environment**
- At each step it perceives a **state**, selects an **action**
- It receives a **reward** and transitions to a new state
- The goal: maximize **cumulative reward** over time

RL is *not* just trial-and-error. Modern RL includes **planning** (using a
model of the environment to simulate and evaluate actions before taking them),
**temporal-difference learning** (bootstrapping value estimates from other
estimates), and **function approximation** (generalizing across states).

The key insight: RL addresses the kind of learning problems that people and
animals face every day -- sequential decisions under uncertainty with long-term
consequences.

### The Two Pillars of RL (from Sutton & Barto)

1. **Prediction**: Learning to estimate the value of states (how much future
   reward to expect from here)
2. **Control**: Learning to select actions that maximize value

### Key Algorithms at a Glance

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| TD(0) | Prediction | Update value toward one-step bootstrap target |
| TD(lambda) | Prediction | Blend TD and Monte Carlo via eligibility traces |
| Q-Learning | Control | Off-policy, learns optimal Q directly |
| SARSA | Control | On-policy, learns Q for current policy |
| Monte Carlo | Both | Wait for episode end, update toward actual return |
| Policy Gradient | Control | Directly optimize policy parameters |
| Dyna | Both | Interleave real experience with model-based planning |
| R-Learning | Control | Average-reward formulation for continuing tasks |

---

## The Bitter Lesson

[The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html)
(Rich Sutton, March 13, 2019) -- one of the most influential essays in AI:

> The biggest lesson that can be read from 70 years of AI research is that
> general methods that leverage computation are ultimately the most effective,
> and by a large margin.

The pattern repeats across chess, Go, speech, vision:

1. Researchers try to build in human knowledge
2. This helps in the short term, is personally satisfying
3. In the long run it plateaus and inhibits progress
4. Breakthrough arrives from an opposing approach: **scaling computation
   through search and learning**

The two methods that scale arbitrarily are **search** and **learning**.

> We want AI agents that can discover like we can, not which contain what we
> have discovered. Building in our discoveries only makes it harder to see how
> the discovering process can be done.

### Implications for Vidya

This challenges our symbolic layer directly. Vidya builds in word validity
constraints, concept coherence, and topic depth penalties. The bitter lesson
says this is the wrong long-term bet -- that we should instead find ways for
the model to *learn* these constraints itself.

However, Vidya's constraints are *structural* (word validity) not *knowledge*
(what words mean), and they're applied at inference time, not baked into the
learned weights. This is closer to the "meta-methods that can find and capture
arbitrary complexity" that Sutton advocates.

The key question: **Can we replace Vidya's hand-crafted symbolic constraints
with learned ones via reinforcement learning?**

---

## Verification: The Key to AI

From [Verification, The Key to AI](http://incompleteideas.net/IncIdeas/Verification.html)
(Rich Sutton, 2001):

> The key to a successful AI is that it can tell for itself whether or not it
> is working correctly.

**The Verification Principle:**
> An AI system can create and maintain knowledge only to the extent that it can
> verify that knowledge itself.

This is exactly what RL provides: a reward signal that the agent uses to verify
its own behavior. Deep Blue verifies moves through search. TD-Gammon verifies
its scoring function through self-play.

### Connection to Vidya

Vidya currently has no way to verify its own output quality. The symbolic
layer enforces *validity* (are words real?) but not *quality* (is the text
meaningful? coherent? interesting?). RL could provide this missing piece:

- **Reward for coherent text**: Use a learned critic to score generated passages
- **Self-play through prompting**: Generate, evaluate, improve
- **Verification through prediction**: TD-style updates to value estimates of
  partial text

---

## Core RL Algorithms in Lisp

Rich Sutton wrote the reference implementations for his RL textbook in Common
Lisp. These are the actual code used to generate the figures in *Reinforcement
Learning: An Introduction* (2nd edition, 2018).

Source: [incompleteideas.net/book/code/code2nd.html](http://incompleteideas.net/book/code/code2nd.html)

---

### Temporal-Difference Learning

**TD Prediction on the Random Walk** (Chapter 6, Example 6.2)

The simplest illustration of TD learning: a random walk on 5 states, where the
agent learns to predict the probability of terminating on the right.

```lisp
;;; TD(lambda) learning on a discrete-state random walk
;;; From Sutton's original 1988 TD paper

(defpackage :discrete-walk
  (:use :common-lisp :g :ut :graph)
  (:nicknames :dwalk))

(in-package :dwalk)

(defvar n 5)                            ; number of nonterminal states
(defvar w)                              ; the vector of weights = predictions
(defvar e)                              ; the eligibility trace
(defvar lambda .9)                      ; trace decay parameter
(defvar alpha 0.1)                      ; learning-rate parameter
(defvar initial-w 0.5)

(defun setup (num-runs num-walks)
  (setq w (make-array n))
  (setq e (make-array n))
  (setq standard-walks (standard-walks num-runs num-walks))
  (length standard-walks))

(defun init ()
  (loop for i below n do (setf (aref w i) initial-w)))

(defun learn (x target)
  "TD update: adjust weight toward target"
  (incf (aref w x) (* alpha (- target (aref w x)))))

(defun process-walk (walk)
  "Process a complete walk using TD(lambda)"
  (destructuring-bind (outcome states) walk
    (init-traces)
    ;; Forward pass: update each state toward the next state's prediction
    (loop for s1 in states
          for s2 in (rest states)
          do (learn s1 (aref w s2)))
    ;; Terminal update: last state updated toward actual outcome
    (learn (first (last states)) outcome)))

(defun random-walk (n &optional (random-state *random-state*))
  "Generate a random walk episode"
  (loop with start-state = (round (/ n 2))
        for x = start-state then (with-prob .5 (+ x 1) (- x 1) random-state)
        while (AND (>= x 0) (< x n))
        collect x into xs
        finally (return (list (if (< x 0) 0 1) xs))))

(defun residual-error ()
  "RMSE between current and correct predictions"
  (rmse 0 (loop for i below n
                when (>= (aref w i) -.1)
                collect (- (aref w i)
                           (/ (+ i 1) (+ n 1))))))
```

**Key insight**: TD learning updates predictions *before* the final outcome
is known, bootstrapping from other predictions. This is more data-efficient
than Monte Carlo methods, which must wait for the episode to end.

---

### Value Iteration & Dynamic Programming

**Gridworld** (Chapter 3, Example 3.5) -- Computing state values via
Bellman equations:

```lisp
(defvar V)
(defvar rows 5)
(defvar columns 5)
(defvar states 25)
(defvar gamma 0.9)

(defun compute-V ()
  "Iterative policy evaluation: compute V under equiprobable random policy"
  (loop for delta = (loop for x below states
                          for old-V = (aref V x)
                          do (setf (aref V x)
                                   (mean (loop for a below 4 collect
                                               (full-backup x a))))
                          sum (abs (- old-V (aref V x))))
        until (< delta 0.000001)))

(defun compute-V* ()
  "Value iteration: compute optimal V* by taking max over actions"
  (loop for delta = (loop for x below states
                          for old-V = (aref V x)
                          do (setf (aref V x)
                                   (loop for a below 4 maximize
                                               (full-backup x a)))
                          sum (abs (- old-V (aref V x))))
        until (< delta 0.000001)))

(defun full-backup (x a)
  "One-step Bellman backup: reward + discounted next-state value"
  (let (r y)
    (cond ((= x AA) (setq r +10) (setq y AAprime))
          ((= x BB) (setq r +5)  (setq y BBprime))
          ((off-grid x a) (setq r -1) (setq y x))
          (t (setq r 0) (setq y (next-state x a))))
    (+ r (* gamma (aref V y)))))
```

**Gambler's Problem** (Chapter 4, Example 4.3) -- Pure value iteration:

```lisp
;;; The gambler wagers on coin flips to reach $100.
;;; State = stake (0-100), Action = bet size

(defvar V (make-array 101 :initial-element 0))
(setf (aref V 100) 1)  ; goal state has value 1
(defvar p .45)          ; probability of heads (less than fair)

(defun backup-action (s a)
  "Expected value of betting a from stake s"
  (+ (* p (aref V (+ s a)))
     (* (- 1 p) (aref V (- s a)))))

(defun vi (&optional (epsilon .00000001))
  "Value iteration to criterion epsilon"
  (loop while (< epsilon
                 (loop for s from 1 below 100
                       for old-V = (aref V s)
                       do (setf (aref V s)
                                (loop for a from 1 upto (min s (- 100 s))
                                      maximize (backup-action s a)))
                       maximize (abs (- old-V (aref V s)))))))

(defun policy (s)
  "Greedy policy: bet the amount that maximizes expected value"
  (loop with best-value = -1
        with best-action
        for a from 1 upto (min s (- 100 s))
        for this-value = (backup-action s a)
        do (when (> this-value (+ best-value .0000000001))
             (setq best-value this-value)
             (setq best-action a))
        finally (return best-action)))
```

**Jack's Car Rental** (Chapter 4, Figure 4.2) -- Policy iteration with
Poisson dynamics:

```lisp
;;; States: (n1, n2) = cars at each location (max 20)
;;; Actions: cars to transfer overnight (-5 to +5)
;;; Dynamics: Poisson requests and returns

(defvar lambda-requests1 3)
(defvar lambda-requests2 4)
(defvar lambda-dropoffs1 3)
(defvar lambda-dropoffs2 2)

(defun poisson (n lambda)
  "Probability of n events under Poisson distribution"
  (* (exp (- lambda))
     (/ (expt lambda n)
        (factorial n))))

(defun policy-eval ()
  "Evaluate current policy until convergence"
  (loop while (< theta
                 (loop for n1 upto 20 maximize
                       (loop for n2 upto 20
                             for old-v = (aref V n1 n2)
                             for a = (aref policy n1 n2)
                             do (setf (aref V n1 n2) (backup-action n1 n2 a))
                             maximize (abs (- old-v (aref V n1 n2))))))))

(defun policy-iteration ()
  "Alternate policy evaluation and greedy improvement"
  (loop for count from 0
        do (policy-eval)
        do (print count)
        while (greedify)))
```

---

### Monte Carlo Methods

**Blackjack** (Chapter 5, Example 5.1) -- First-visit MC prediction:

```lisp
;;; State: (dealer-card, player-count, usable-ace?)
;;; Actions: hit (1) or stick (0)

(defun episode ()
  "Play one episode, collecting state visits"
  (let (dc-hidden pcard1 pcard2 outcome)
    (setq episode nil)
    (setq dc-hidden (card))
    (setq dc (card))
    (setq pcard1 (card))
    (setq pcard2 (card))
    (setq ace (OR (= 1 pcard1) (= 1 pcard2)))
    (setq pc (+ pcard1 pcard2))
    (if ace (incf pc 10))
    (unless (= pc 21)                   ; natural blackjack
      (loop do (push (list dc pc ace) episode)
            while (= 1 (aref policy dc pc (if ace 1 0)))
            do (draw-card)
            until (bust?)))
    (setq outcome (outcome dc dc-hidden))
    (learn episode outcome)
    outcome))

(defun learn (episode outcome)
  "Incremental mean update for each visited state"
  (loop for (dc pc ace-boolean) in episode
        for ace = (if ace-boolean 1 0) do
        (when (> pc 11)
          (incf (aref N dc pc ace))
          (incf (aref V dc pc ace)
                (/ (- outcome (aref V dc pc ace))
                   (aref N dc pc ace))))))
```

**Monte Carlo ES (Exploring Starts)** -- with policy improvement:

```lisp
(defun learn (episode outcome)
  "MC-ES: update Q-values and greedify policy"
  (loop for (dc pc ace-boolean action) in episode
        for ace = (if ace-boolean 1 0) do
        (when (> pc 11)
          (incf (aref N dc pc ace action))
          (incf (aref Q dc pc ace action)
                (/ (- outcome (aref Q dc pc ace action))
                   (aref N dc pc ace action)))
          ;; Greedy policy improvement
          (let ((policy-action (aref policy dc pc ace))
                (other-action (- 1 policy-action)))
            (when (> (aref Q dc pc ace other-action)
                     (aref Q dc pc ace policy-action))
              (setf (aref policy dc pc ace) other-action))))))
```

---

### Q-Learning & Double Q-Learning

**Double Q-Learning** (Chapter 6, Example 6.7) -- Eliminates maximization
bias by maintaining two independent Q-value estimates:

```lisp
;;; The maximization bias problem:
;;; State A: "right" gives -1 (optimal), "left" goes to B with -1.1
;;; State B: 10 actions, all 0 mean but variance 1
;;; Q-learning overestimates B's value due to max over noisy estimates

(defparameter Q (make-array (list 2 10)))  ; Two independent estimators
(defparameter Qright (make-array 2))
(defparameter Qwrong (make-array 2))

(defun go-left ()
  "Double Q-learning update: use one estimator to select, other to evaluate"
  (let ((e (random 2)))  ; randomly pick which estimator to update
    (incf (aref Qwrong e)
          (* alpha (+ 0.0
                      (aref Q (- 1 e) (argmax-single e))  ; evaluate with OTHER
                      (- (aref Qwrong e))))))
  ;; Also update the B-state estimates
  (let ((a (argmax-double)))
    (when (< (random 1.0) epsilon) (setq a (random 10)))
    (let ((e (random 2)))
      (incf (aref Q e a)
            (* alpha (- (random-normal) .1 (aref Q e a)))))))

(defun argmax-double ()
  "Select action by summing BOTH estimators (reduces bias)"
  (loop with best-args = (list 0)
        with best-value = (+ (aref Q 0 0) (aref Q 1 0))
        for i from 1 below 10
        for value = (+ (aref Q 0 i) (aref Q 1 i))
        do (cond ((> value best-value)
                  (setq best-value value)
                  (setq best-args (list i)))
                 ((= value best-value)
                  (push i best-args)))
        finally (return (nth (random (length best-args)) best-args))))
```

---

### Multi-Armed Bandits

**10-Armed Testbed** (Chapter 2, Figure 2.1) -- The simplest RL problem:

```lisp
(defvar n 10)       ; number of arms
(defvar Q*)         ; true action values (unknown to agent)
(defvar Q)          ; agent's estimates
(defvar n_a)        ; count of times each action taken

(defun learn (a r)
  "Sample-average update: Q(a) += (r - Q(a)) / N(a)"
  (incf (aref n_a a))
  (incf (aref Q a) (/ (- r (aref Q a))
                      (aref n_a a))))

(defun epsilon-greedy (epsilon)
  "With probability epsilon explore randomly, else exploit best known"
  (with-prob epsilon
    (random n)
    (arg-max-random-tiebreak Q)))
```

**UCB (Upper Confidence Bound)** (Chapter 2, Figure 2.4):

```lisp
(defun UCB-selection (time)
  "Select action maximizing Q(a) + c * sqrt(ln(t) / N(a))"
  (loop for a below n
    with lnt = (log time)
    do (setf (aref Qtemp a)
             (+ (aref Q a) (* c (sqrt (/ lnt (aref n_a a))))))
    finally (return (arg-max-random-tiebreak Qtemp))))
```

**Optimistic Initial Values** (Chapter 2, Figure 2.3) -- Encourage early
exploration by initializing Q high:

```lisp
(defvar alpha 0.1)  ; constant step-size (not sample average)

(defun init ()
  (loop for a below n do
        (setf (aref Q a) Q0)   ; Q0 = 5.0 encourages exploration
        (setf (aref n_a a) 0)))

(defun learn (a r)
  "Constant step-size update (recency-weighted average)"
  (incf (aref Q a) (* alpha (- r (aref Q a)))))
```

**Softmax Action Selection** (Chapter 2, Exercise 2.2):

```lisp
(defun policy (temperature)
  "Softmax: probability proportional to exp(Q(a) / tau)"
  (loop for a below n
        for value = (aref Q a)
        sum (exp (/ value temperature)) into total-sum
        collect total-sum into partial-sums
        finally (return
                 (loop with rand = (random (float total-sum))
                       for partial-sum in partial-sums
                       for a from 0
                       until (> partial-sum rand)
                       finally (return a)))))
```

---

### Gradient Bandits

**Gradient Bandit Algorithm** (Chapter 2, Figure 2.5) -- Learn action
*preferences* H(a) rather than value estimates, using a softmax policy
and stochastic gradient ascent:

```lisp
(defvar H)          ; preference for each action
(defvar Rbar)       ; baseline: running average of rewards
(defvar policy)     ; softmax probabilities

(defun learn (A R time-step)
  "Gradient update: increase preference for actions that beat baseline"
  (incf Rbar (/ (- R Rbar) (1+ time-step)))  ; update baseline
  (let ((alpha-delta (* alpha (- R Rbar))))
    ;; Decrease preference for all actions proportional to their probability
    (loop for a below n do
          (decf (aref H a) (* alpha-delta (aref policy a))))
    ;; Increase preference for the taken action
    (incf (aref H A) alpha-delta)))
```

This is the simplest form of **policy gradient** -- the same principle that
scales to deep RL (REINFORCE, PPO, etc.).

---

### R-Learning

**Access-Control Queuing Task** (Chapter 10, Example 10.2) -- Average-reward
RL for continuing (non-episodic) tasks:

```lisp
;;; N servers, M priority levels. Agent decides: accept (1) or reject (0)?

(defvar rho)  ; estimate of average reward rate

(defun R-learning (steps)
  "R-learning: differential Q-learning for average-reward setting"
  (loop repeat steps
        for s = (+ num-free-servers (* priority N+1)) then s-prime
        for a = (with-prob epsilon (random 2)
                  (if (> (aref Q s 0) (aref Q s 1)) 0 1))
        for r = (if (AND (= a 1) (> num-free-servers 0))
                  (aref reward priority)
                  0)
        ;; ... environment transition ...
        ;; Differential TD update: r - rho + max Q(s') - Q(s,a)
        do (incf (aref Q s a)
                 (* alpha (+ r (- rho)
                             (max (aref Q s-prime 0)
                                  (aref Q s-prime 1))
                             (- (aref Q s a)))))
        ;; Update average reward estimate
        do (when (= (aref Q s a) (max (aref Q s 0) (aref Q s 1)))
             (incf rho (* beta (+ r (- rho)
                                  (max (aref Q s-prime 0)
                                       (aref Q s-prime 1))
                                  (- (max (aref Q s 0) (aref Q s 1)))))))))
```

**Why this matters for Vidya**: Vidya generates text in a *continuing* fashion
(no natural episode boundaries). Average-reward RL (R-learning) is designed
exactly for this setting -- it doesn't require discounting or episode
termination.

---

### Dyna Architecture

**Dyna-AHC** -- The crucial insight of combining real experience with
model-based planning:

```lisp
;;; Dyna: For each real step, do N model-based planning steps

(defvar num-model-steps 10)

(defun learn ()
  "AHC (Actor-Critic) learning update"
  (setq e (aref v x))
  (setq e-prime (+ r (* gamma (aref v y))))
  (incf (aref v x) (* beta (- e-prime e)))    ; critic update
  (incf (aref w x a) (* alpha (- e-prime e)))) ; actor update

(defun learn-model ()
  "Record observed transition in model"
  (setf (aref model-next-state x a) y)
  (setf (aref model-reward x a) r))

(defun trial ()
  "One trial with interleaved real and model-based steps"
  (setq current-state start-state)
  (loop while (not (= current-state goal-state)) do
    ;; Model-based planning: replay past experiences
    (loop repeat num-model-steps do
      (setq x (pick-from-visited-states))
      (setq a (action w x))
      (setq y (aref model-next-state x a))
      (setq r (aref model-reward x a))
      when y do (learn))
    ;; Real experience
    (setq x current-state)
    (setq a (action w x))
    (setq y (aref real-next-state x a))
    (setq r (if (= y goal-state) 1 0))
    (learn)
    (learn-model)
    (setq current-state y)))
```

**Why this matters for Vidya**: Dyna shows that an agent can learn from both
real interaction AND from replaying/simulating past experiences. Vidya's Forth
knowledge layer already stores structured information about the corpus. A Dyna-
like architecture could let Vidya:
- Learn token-level value estimates from real generation
- Use the Forth knowledge to *simulate* likely continuations
- Plan ahead before committing to tokens

---

### TD Model of Classical Conditioning

This is particularly relevant -- it shows TD learning applied to
*prediction of future stimuli*, not just reward maximization:

```lisp
;;; TD model of Pavlovian reinforcement
;;; From Sutton & Barto (1990)

(defvar alpha 0.1)
(defvar gamma 0.95)
(defvar delta 0.2)    ; trace decay for stimulus representation

(defun Vbar (V X)
  "Prediction: weighted sum of associative strengths"
  (max 0 (loop for V-i in V
               for X-i in X
               sum (* V-i X-i))))

(defun steps (num-steps X lambda)
  "Run TD model: X = conditioned stimuli, lambda = unconditioned stimulus"
  (loop repeat num-steps
        for new-Vbar = (Vbar V X)
        for alpha-beta-error = (* alpha beta
                                   (+ lambda
                                      (* gamma new-Vbar)
                                      (- old-Vbar)))
        do (loop for i below n
                 for X-i in X
                 for trace-i in trace
                 do (incf (nth i V) (* alpha-beta-error trace-i))
                 do (incf (nth i trace) (* delta (- X-i trace-i))))
        do (setq old-Vbar (Vbar V X))))
```

**Connection to Vidya**: This model learns *temporal associations* between
stimuli -- exactly what Vidya's concept coherence layer does with static
co-occurrence statistics. A TD-based approach could learn these associations
*online* from the generation process itself.

---

### Mountain Car with Tile Coding

**n-step Sarsa on Mountain Car** (Chapter 10, Figures 10.2-4) -- RL with
function approximation using tile coding:

```lisp
;;; State: (position, velocity), 3 actions: left/none/right
;;; Tile coding: 8 overlapping tilings over the continuous state space

(defparameter num-tilings 8)
(defparameter max-tiles 4096)

(defun mcar-tiles (s a)
  "Map continuous state + action to active tile indices"
  (destructuring-bind (x . xdot) s
    (tiles iht num-tilings
           (list (/ (* x 8) (- 0.5 -1.2))
                 (/ (* xdot 8) (- 0.07 -0.07)))
           (list a))))

(defun q (s a)
  "Q-value = sum of weights at active tiles"
  (loop for i in (mcar-tiles s a) sum (aref theta i)))

(defun episode ()
  "One episode of n-step Sarsa with tile coding"
  (setf (aref Sstore 0) (mcar-init))
  (setf (aref Astore 0) (egreedy (aref Sstore 0)))
  (loop with capT = 1000000
    for tt from 0
    for tau = (+ tt (- n) 1)
    ;; Collect experience
    when (< tt capT)
    do (multiple-value-bind (R Sprime) (mcar-sample (aref Sstore tt) (aref Astore tt))
         (setf (aref Rstore (+ tt 1)) R)
         (if (terminalp Sprime)
           (setq capT (+ tt 1))
           (progn (setf (aref Sstore (+ tt 1)) Sprime)
                  (setf (aref Astore (+ tt 1)) (egreedy Sprime)))))
    ;; n-step return and update
    when (>= tau 0)
    do (loop
         with G = (+ (loop for k from (+ tau 1) to (min (+ tau n) capT)
                           sum (aref Rstore k))
                     (if (< (+ tau n) capT)
                       (q (aref Sstore (+ tau n)) (aref Astore (+ tau n)))
                       0))
         with tiles = (mcar-tiles (aref Sstore tau) (aref Astore tau))
         with alpha-error = (* alpha (- G (q-from-tiles tiles)))
         for i in tiles do (incf (aref theta i) alpha-error))
    until (= tau (- capT 1))))
```

**Sarsa(lambda) on Mountain Car** -- with eligibility traces:

```lisp
(defun episode ()
  "Sarsa(lambda) with replacing traces and tile coding"
  (loop with delta
    initially (fill e 0.0)
    for S = (mcar-init) then Sprime
    for A = (egreedy S) then Aprime
    for tiles = (mcar-tiles S A)
    for (R Sprime) = (mcar-sample S A)
    for Aprime = (unless (terminalp Sprime) (egreedy Sprime))
    do
    ;; Set traces to 1 for active tiles (replacing traces)
    (loop for tile in tiles do (setf (aref e tile) 1))
    ;; Compute TD error
    (setq delta (- R (q-from-tiles tiles)))
    (setq tiles (mcar-tiles Sprime Aprime))
    (setq delta (+ delta (* gamma (q-from-tiles tiles))))
    ;; Update all weights proportional to their trace
    (loop for i below max-tiles
          with alpha-delta = (* alpha delta) do
          (incf (aref theta i) (* alpha-delta (aref e i)))
          (setf (aref e i) (* gamma lambda (aref e i))))
    sum R into Rsum
    until (terminalp Sprime)
    finally (return Rsum)))
```

---

### Tic-Tac-Toe

**Value-based self-play** (Chapter 1) -- The introductory example showing
the core RL loop:

```lisp
;;; States mapped to a value table via index
;;; Magic square positions: 2 9 4 / 7 5 3 / 6 1 8
;;; Three positions summing to 15 = three in a row

(defvar alpha 0.5)    ; learning rate
(defvar epsilon 0.01) ; exploration rate

(defun greedy-move (player state)
  "Select the move leading to the highest-valued successor state"
  (loop with best-value = -1
        for move in (possible-moves state)
        for move-value = (value (next-state player state move))
        when (> move-value best-value)
        do (setf best-value move-value
                 best-move move)
        finally (return best-move)))

(defun update (state new-state)
  "TD(0) value update: V(s) += alpha * [V(s') - V(s)]"
  (set-value state (+ (value state)
                      (* alpha
                         (- (value new-state)
                            (value state))))))

(defun game ()
  "One game: X plays randomly, O learns via TD"
  (setq state initial-state)
  (loop for new-state = (next-state :X state (random-move state))
        for exploratory? = (< (random 1.0) epsilon)
        do (when (terminal-state-p new-state)
             (update state new-state)
             (return (value new-state)))
        (setf new-state (next-state :O new-state
                          (if exploratory?
                            (random-move new-state)
                            (greedy-move :O new-state))))
        (unless exploratory? (update state new-state))
        (when (terminal-state-p new-state) (return (value new-state)))
        (setq state new-state)))
```

---

## Lisp as an AI Substrate

Sutton chose Common Lisp for all his reference implementations. This is not
accidental. Lisp has properties that make it uniquely suited to AI research:

### Why Sutton Uses Lisp

1. **Homoiconicity**: Code is data. Programs can manipulate other programs.
   This enables meta-learning and self-modifying agents.

2. **Interactive development**: REPL-driven exploration. Run an experiment,
   inspect state, modify parameters, continue -- without restarting.

3. **Symbolic + numeric**: First-class support for both symbolic manipulation
   (lists, trees, pattern matching) and numerical computation (arrays, floats).

4. **Macros**: Define new control structures that look like language primitives.
   Sutton's `with-prob` macro is a perfect example.

5. **Dynamic typing with optional declarations**: Rapid prototyping with the
   option to add type declarations for performance.

### Lisp Patterns in Sutton's Code

Recurring patterns that reveal design principles:

```lisp
;; Pattern 1: Incremental mean update (appears everywhere)
(incf (aref Q a) (/ (- r (aref Q a)) (aref n_a a)))

;; Pattern 2: Epsilon-greedy with macro
(with-prob epsilon (random n) (arg-max-random-tiebreak Q))

;; Pattern 3: Tabular representation via arrays
(defvar V (make-array states :initial-element 0.0))
(defvar Q (make-array (list states actions) :initial-element 0.0))

;; Pattern 4: Episode collection via loop
(loop for state = start then next-state
      collect state
      until terminal-p
      finally (return (list outcome states)))

;; Pattern 5: Multi-run averaging
(multi-mean (loop repeat num-runs collect (run num-episodes)))
```

### Connection to Vidya's Forth

Vidya already uses a Forth interpreter as its symbolic substrate. Forth and
Lisp share key properties:
- **Stack-based evaluation** (Forth) vs **tree-based evaluation** (Lisp)
- Both support **extensibility** through new word/function definitions
- Both are **minimal** yet **complete**

The question: should Vidya's knowledge layer speak Lisp instead of (or in
addition to) Forth? Lisp's list processing and recursion could enable:
- **Recursive concept hierarchies** (not just flat associations)
- **Pattern matching** on generated text
- **Rule-based reasoning** that complements neural predictions

Or: should Vidya's Forth grow Lisp-like features (cons cells, recursion)?

---

## Forth vs Lisp: Why Forth Wins for Vidya

Sutton wrote all his RL reference implementations in Common Lisp, and Lisp has
genuine strengths for AI research. But for Vidya's symbolic substrate, **Forth
has deeper power than Lisp**. The decision: keep Forth, grow Lisp-like features
into it, rather than switching.

### Forth is Concatenative

Any Forth program can be split at any point and both halves are valid programs.
You compose by juxtaposition. Lisp doesn't have this -- you can't split an
s-expression arbitrarily. This matters for Vidya because **token generation is
concatenative**. Tokens arrive left-to-right, one at a time. Forth's evaluation
order *is* the generation order. Lisp evaluates inside-out, which fights the
temporal structure of text.

### The Stack is a Natural Attention Mechanism

What's on top of the stack is what's being attended to right now. Push = focus.
Pop = release. This maps cleanly onto Vidya's concept activation window (the
16-token decay). Lisp's nested lexical scoping doesn't have this temporal
quality. The stack *is* a working memory with recency bias -- exactly the
cognitive model that Vidya's symbolic layer implements.

### Forth's Dictionary IS the Knowledge Graph

Words defined in terms of other words isn't a data structure *representing*
knowledge -- it's knowledge that *computes*. Vidya's Concept entries with
associations are already halfway there. A Lisp association list is inert data
that needs external code to interpret it. A Forth word *does something* when
you invoke it. The dictionary is simultaneously a namespace, a knowledge base,
and an executable program.

### Verifiability

Vidya already does stack-effect validation in O(n). This connects directly to
Sutton's Verification Principle -- the AI can tell for itself whether a word
definition is valid. Lisp's general recursion makes this undecidable. Forth's
constraint: every word must consume and produce a known number of stack items.
This is *exactly* the kind of structural constraint that survives the Bitter
Lesson -- it's not domain knowledge, it's a meta-method.

### Minimality is the Point

Forth isn't less powerful than Lisp -- it's more *primitive*, in the
mathematical sense. Closer to the computation itself. You can build up from
Forth to whatever you need. Building down from Lisp to the stack machine is
fighting gravity. Forth's entire semantics fit in a paragraph: there's a
dictionary, a data stack, a return stack, and words consume input and execute.
Everything else is defined in terms of these primitives.

### What to Grow from Lisp

The useful features of Lisp can be added to Forth without abandoning Forth's
nature:

| Lisp Feature | Forth Implementation |
|---|---|
| **Cons cells / pairs** | Push structured pairs onto the stack. Get lists and trees without leaving Forth. |
| **Recursive definitions** | Use the return stack (which Forth already has). Add a base-case check idiom. |
| **Pattern matching** | A `MATCH` word that destructures stack values. This is where Lisp's real power lives, and it fits naturally in Forth's vocabulary. |
| **Quotations / lambdas** | Push a code block onto the stack as data, execute it later. Factor (a modern Forth descendant) does this beautifully. Gives you anonymous functions without s-expressions. |
| **Map / filter / reduce** | Quotations + stack combinators. `' square EACH` instead of `(mapcar #'square list)`. |

### What Lisp Has That We Deliberately Leave Out

**Full homoiconicity** -- code as data in the most general sense. Quotations
get you 90% of the way there. The remaining 10% (full metaprogramming, eval of
arbitrary constructed code) is arguably *dangerous* for an AI that needs to be
verifiable. Sutton's Verification Principle says the AI must be able to check
its own knowledge. Unrestricted self-modification undermines this.

### The Deeper Argument

Forth's evaluation model matches the causal structure of time. Things happen in
sequence. Effects follow causes. The stack accumulates context and releases it.
This is how text works, how thought works, how reinforcement learning works --
one step at a time, left to right, with a finite working memory that decays.

Lisp's tree model is spatially elegant but temporally unnatural. It requires
knowing the whole expression before evaluating any of it. Forth starts
executing the moment the first word arrives. For an AI that generates tokens
one at a time and must evaluate them incrementally, this is the right
foundation.

---

## Applying RL to Vidya

### Current Architecture (No RL)

Vidya is a neurosymbolic language model:
- **Neural**: 6-layer GPT-2 style transformer (1.25M params, RoPE, KV-cache)
- **Symbolic**: 5 constraints applied to logits before sampling
- **Knowledge**: Forth dictionary of concepts extracted from corpus statistics

The neural model learns via standard **supervised learning** (cross-entropy
loss, Adam optimizer, cosine schedule). The symbolic layer uses **hand-crafted
rules**. There is no reinforcement learning.

### Where RL Could Enter

#### 1. Reward-Guided Generation (RLHF-Lite)

Instead of sampling from constrained logits, use RL to learn which tokens
produce good text:

```
State:   partial text generated so far
Action:  next token to emit
Reward:  quality signal (coherence, validity, novelty)
```

The reward could come from:
- **Self-evaluation**: Does the generated text activate diverse concepts?
- **Prediction accuracy**: Can the model predict what comes next in the corpus?
- **Forth verification**: Does the generated text form valid Forth programs?

#### 2. Learned Constraints (Replace Symbolic Layer)

Each of Vidya's 5 constraints could be *learned* rather than hand-coded:

| Current Constraint | RL Replacement |
|---|---|
| Repetition penalty (-1.5) | Learn penalty magnitude from reward signal |
| Word boundary detection | Learn tokenization preferences |
| Word validation (hard mask) | Learn soft validity preferences |
| Concept coherence (+2.0 boost) | Learn association strengths via TD |
| Topic depth penalty | Learn exploration-exploitation balance |

The TD model of classical conditioning (above) is directly applicable:
concept activation is a *conditioned stimulus*, and text quality is the
*unconditioned stimulus*. TD learning would let Vidya learn which concept
activations predict good text.

#### 3. Dyna-Style Planning

Vidya could use its Forth knowledge as a *world model*:

```
Real experience:  Generate token, observe reward
Model experience: Use Forth associations to simulate likely continuations
Planning:         Evaluate multiple token sequences before committing
```

This is exactly Dyna: interleave real generation with model-based look-ahead.

#### 4. Average-Reward Formulation

Text generation is a *continuing task* -- there's no natural episode boundary.
R-learning (shown above) is designed for exactly this:

```
rho = running estimate of average text quality
TD error = r - rho + V(s') - V(s)
```

This avoids the need for discounting (which biases toward short-term quality)
and naturally handles the fact that text generation doesn't have "episodes."

#### 5. Multi-Armed Bandit for Token Selection

At each generation step, Vidya faces a multi-armed bandit problem:
- ~580 token "arms" (the BPE vocabulary)
- Reward = text quality after choosing that token
- Exploration-exploitation tradeoff

UCB or gradient bandit methods could replace or supplement the current
softmax sampling:

```
score(token) = neural_logit(token) + c * sqrt(ln(t) / N(token))
```

#### 6. Eligibility Traces for Credit Assignment

When text quality is measured at the sentence level, which *token* deserves
credit? Eligibility traces solve this:

```
e(token) *= gamma * lambda    ; decay all traces
e(chosen_token) = 1           ; mark the chosen token
theta += alpha * delta * e    ; update proportional to trace
```

This is exactly what TD(lambda) with replacing traces does (Mountain Car
example above).

### Proposed Architecture: Vidya-RL

```
Corpus --> BPE Training --> Tokenizer
                               |
                    +----------+----------+
                    |                     |
              Neural Training      Knowledge Extraction
              (supervised)         (corpus statistics)
                    |                     |
                    v                     v
              Transformer          Forth Dictionary
              (6L, 128d)          (concepts, assoc.)
                    |                     |
                    +----------+----------+
                               |
                         Generation Loop:
                    1. Forward pass -> logits
                    2. Symbolic constraints -> constrained logits
                    3. RL value adjustment -> final logits    <-- NEW
                    4. Softmax, sample token
                    5. Compute reward signal                  <-- NEW
                    6. TD update to value estimates            <-- NEW
                    7. Update Forth knowledge if needed        <-- NEW
                               |
                         Output text
```

---

## Key References & Links

### The Textbook

- **Reinforcement Learning: An Introduction** (2nd ed., 2018)
  by Sutton & Barto
  - [Full PDF](http://incompleteideas.net/book/RLbook2020.pdf)
  - [Code (all Lisp examples)](http://incompleteideas.net/book/code/code2nd.html)
  - [Python reimplementations](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - [Julia reimplementations](https://github.com/Ju-jl/ReinforcementLearningAnIntroduction.jl)

### Key Essays

- [The Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) (2019)
- [Verification, The Key to AI](http://incompleteideas.net/IncIdeas/Verification.html) (2001)

### Research Proposals & Papers

- [The Alberta Plan for AI Research](https://arxiv.org/pdf/2208.11173.pdf)
- [Top 10 Readings for RL Approach to AI](https://docs.google.com/document/d/1juudZLXpqMsuAXg7zGFlkRdBf8hffDzSChWkHAmJci0/edit)
- [2025 CCAI Chair Proposal](http://incompleteideas.net/CCAIprop2025.pdf)
- [ACM Turing Award Video](https://www.youtube.com/watch?v=RrXibq7-W6o)

### Courses

- [RL MOOC (Coursera)](https://www.coursera.org/specializations/reinforcement-learning)
- [CMPUT 609 - RL II](https://drive.google.com/drive/folders/0B3w765rOKuKANmxNbXdwaE1YU1k?usp=sharing)

### Software

- [Tile Coding v3](http://incompleteideas.net/tiles/tiles3.html) -- Function approximation
- [TD Model (Lisp)](http://incompleteideas.net/TDmodel.lisp) -- Classical conditioning
- [Dyna-AHC (Lisp)](http://incompleteideas.net/simple-dyna-ahc.lisp) -- Planning + learning
- [Acrobot (Lisp)](http://incompleteideas.net/book/code/acrobot.lisp) -- Control problem
- [G Graphics for MCL](http://incompleteideas.net/G/g.html) -- Sutton's Lisp graphing package

### Publications Highlights

- [RL FAQ](http://incompleteideas.net/RL-FAQ.html)
- [Publications page](http://incompleteideas.net/publications.html) with highlights:
  - Welcome to the Era of Experience
  - Loss of Plasticity and Continual Backprop (Nature)
  - The STOMP progression (SwiftTD, Swift-Sarsa)
  - The Big World Hypothesis
  - Reward centering
  - The common model of the intelligent agent
  - Horde, nexting, and predictive knowledge
  - Temporal-difference learning (original)
  - TD model of Pavlovian conditioning
  - Dyna and its extensions
  - Options framework (temporal abstraction)
  - Policy gradient methods
  - PSRs (Predictive State Representations)

### Classic Papers (hosted by Sutton)

- [Minsky, 1960, Steps to AI](http://incompleteideas.net/papers/Minsky60steps.pdf)
- [Samuel, 1959 (checkers)](http://incompleteideas.net/papers/samuel.pdf)
- [Watkins thesis (Q-learning)](http://incompleteideas.net/papers/watkins-thesis.pdf)
- [Tesauro, 1992 (TD-Gammon)](http://incompleteideas.net/papers/tesauro-92.pdf)
- [Williams, 1992 (REINFORCE)](http://incompleteideas.net/papers/williams-92.pdf)
- [Selfridge, 1958 (Pandemonium)](http://incompleteideas.net/papers/pandemonium.pdf)
- [Good, 1965 (Ultraintelligent Machine)](http://incompleteideas.net/papers/Good65ultraintelligent.pdf)
- [The Hedonistic Neuron (Klopf)](http://incompleteideas.net/papers/The_Hedonistic_Neuron.pdf)

### All Lisp Code Files (from the textbook)

| File | Chapter | Topic |
|------|---------|-------|
| [TTT.lisp](http://incompleteideas.net/book/code/TTT.lisp) | 1 | Tic-Tac-Toe (value-based self-play) |
| [testbed.lisp](http://incompleteideas.net/book/code/testbed.lisp) | 2 | 10-armed bandit testbed |
| [softmax.lisp](http://incompleteideas.net/book/code/softmax.lisp) | 2 | Softmax action selection |
| [optimistic.lisp](http://incompleteideas.net/book/code/optimistic.lisp) | 2 | Optimistic initial values |
| [UCB.lisp](http://incompleteideas.net/book/code/UCB.lisp) | 2 | Upper Confidence Bound |
| [gradbandits.lisp](http://incompleteideas.net/book/code/gradbandits.lisp) | 2 | Gradient bandits |
| [summary.lisp](http://incompleteideas.net/book/code/summary.lisp) | 2 | Multi-algorithm comparison |
| [gridworld5x5.lisp](http://incompleteideas.net/book/code/gridworld5x5.lisp) | 3 | Gridworld (value functions) |
| [gridworld4x4.lisp](http://incompleteideas.net/book/code/gridworld4x4.lisp) | 4 | Policy evaluation |
| [jacks.lisp](http://incompleteideas.net/book/code/jacks.lisp) | 4 | Jack's Car Rental (policy iteration) |
| [gambler.lisp](http://incompleteideas.net/book/code/gambler.lisp) | 4 | Gambler's Problem (value iteration) |
| [blackjack1.lisp](http://incompleteideas.net/book/code/blackjack1.lisp) | 5 | Blackjack (MC prediction) |
| [blackjack2.lisp](http://incompleteideas.net/book/code/blackjack2.lisp) | 5 | Blackjack (MC ES) |
| [blackjack3-rollout-one-state.lisp](http://incompleteideas.net/book/code/blackjack3-rollout-one-state.lisp) | 5 | Blackjack (single state) |
| [InfinteVariance.lisp](http://incompleteideas.net/book/code/InfinteVariance.lisp) | 5 | Infinite variance example |
| [walk.lisp](http://incompleteideas.net/book/code/walk.lisp) | 6 | TD Random Walk |
| [walk-batch.lisp](http://incompleteideas.net/book/code/walk-batch.lisp) | 6 | Batch TD Random Walk |
| [doubleQ.lisp](http://incompleteideas.net/book/code/doubleQ.lisp) | 6 | Double Q-learning |
| [singleQ.lisp](http://incompleteideas.net/book/code/singleQ.lisp) | 6 | Conventional Q-learning |
| [online.lisp](http://incompleteideas.net/book/code/online.lisp) | 7,12 | Online n-step TD / TD(lambda) |
| [offline.lisp](http://incompleteideas.net/book/code/offline.lisp) | 7,12 | Offline n-step TD / lambda-return |
| [sampling2.lisp](http://incompleteideas.net/book/code/sampling2.lisp) | 8 | Trajectory sampling |
| [aggreg-walknew.lisp](http://incompleteideas.net/book/code/aggreg-walknew.lisp) | 9 | State aggregation |
| [generalization.lisp](http://incompleteideas.net/book/code/generalization.lisp) | 9 | Coarse coding |
| [FAwalknew.lisp](http://incompleteideas.net/book/code/FAwalknew.lisp) | 9 | Function approximation |
| [mcar-nstep.lisp](http://incompleteideas.net/book/code/mcar-nstep.lisp) | 10 | Mountain Car (n-step Sarsa) |
| [queuing.lisp](http://incompleteideas.net/book/code/queuing.lisp) | 10 | R-learning (average reward) |
| [baird-continuing2.lisp](http://incompleteideas.net/book/code/baird-continuing2.lisp) | 11 | Baird counterexample |
| [mcar-sarsa-lambda.lisp](http://incompleteideas.net/book/code/mcar-sarsa-lambda.lisp) | 12 | Mountain Car Sarsa(lambda) |

### Additional Lisp Code (from software page)

| File | Topic |
|------|-------|
| [TDmodel.lisp](http://incompleteideas.net/TDmodel.lisp) | TD model of classical conditioning |
| [simple-dyna-ahc.lisp](http://incompleteideas.net/simple-dyna-ahc.lisp) | Dyna-AHC (planning + learning) |
| [acrobot.lisp](http://incompleteideas.net/book/code/acrobot.lisp) | Acrobot control problem |

---

*Compiled from [incompleteideas.net](http://incompleteideas.net/), the website
of Rich Sutton, co-author of the RL textbook and co-recipient of the 2025
ACM Turing Award for foundational contributions to reinforcement learning.*
