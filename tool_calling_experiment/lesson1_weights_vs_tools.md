# Lesson 1: Weights vs. Tools -- What Should a Model Know vs. Look Up?

A systematic analysis of which knowledge components in autonomous vehicle scene
classification belong in model weights, which belong in external tools, and what
this decomposition reveals about the tool-calling paradigm.

---

## Section 1: Scene Classification

Scene classification requires the model to look at a dashcam image and output
one of five labels: nominal, flooded, incident_zone, mounted_police, flagger.
Below is every knowledge component needed to get this right, categorized by
where that knowledge should live.

### 1.1 Visual Appearance of Each Scene Type

**Category: Perceptual (weights)**

The model must recognize what each scene looks like from pixels alone:

- **nominal**: Normal driving conditions. Clear road, standard traffic, no
  unusual objects. The most visually diverse category (78.1% of data) --
  suburban streets, highways, urban intersections, night driving, rain, etc.
- **flooded**: Standing water on the road surface. May range from shallow
  puddles to road-covering floods. Key visual cue: reflective water surface
  disrupting normal road texture.
- **incident_zone**: Active emergency scene. Emergency vehicles with lights,
  crashed vehicles, road closures, police tape, fire trucks. The critical
  feature is *active incident response*, not just the presence of roadside
  objects.
- **mounted_police**: Officers on horseback. The distinguishing feature is
  literally a horse on or near the road. Relatively unambiguous once you see it.
- **flagger**: A human directing traffic, typically at a construction zone.
  The person is usually wearing high-visibility clothing and holding a stop/slow
  sign or flag.

**Why weights**: No function call can look at an image and report what it sees.
Visual recognition is the irreducible core of this task. A tool that says
"flooded roads have standing water" is useless unless the model can already
detect standing water. This is a closed-loop dependency: the model needs
perception to call the right tool, and the tool cannot provide perception.

### 1.2 Distinguishing nominal_triggers from Real Incidents

**Category: Perceptual (weights) -- and this is where the model fails**

The dataset contains a sub-population called `nominal_triggers`: nominal scenes
that contain visual elements commonly associated with incidents -- traffic
cones, barriers, construction equipment, parked emergency vehicles (not
actively responding), road work signage. There are 2,801 of these samples, and
the fine-tuned model misclassifies them as incident_zone at an 81% rate.

The distinction is subtle and entirely perceptual:

| Visual Element | nominal_triggers | incident_zone |
|---|---|---|
| Traffic cones | Present (lane guidance) | Present (accident perimeter) |
| Barriers | Present (construction) | Present (road closure) |
| Emergency vehicles | Absent or parked/inactive | Present with lights on |
| Road closure | No | Yes, active |
| Debris/damage | No | Often yes |
| Human activity | Construction workers | First responders |

**Why weights**: The difference between "cones marking a construction lane
shift" and "cones around a crash scene" is a perceptual judgment. No statistical
tool can tell you which one you are looking at. This is not a question of base
rates -- it is a question of what the image contains. The model must learn that
the *context* of these objects matters, not just their presence.

This is the single most important knowledge component in the entire task, and it
is purely perceptual. The fine-tuned model has learned a shortcut: "see cone ->
predict incident_zone." No tool can undo this shortcut. Only retraining (or a
second model that can actually see the image) can fix it.

### 1.3 Base Rates of Each Class

**Category: Statistical (tool)**

The ground truth distribution:

| Class | Rate |
|---|---|
| nominal | 78.1% |
| flagger | 7.7% |
| flooded | 7.2% |
| incident_zone | 3.7% |
| mounted_police | 3.2% |

This is exactly the kind of information a tool should provide. The model
predicted incident_zone at 46.8% -- a 12.5x over-prediction relative to the
true 3.7% base rate. If a model calls `check_scene_prior("incident_zone")` and
gets back "base rate is 3.7%, you are predicting this at 46.8%," that is
actionable information.

**Why tool**: Base rates are static facts derived from the training distribution.
They change when the deployment distribution changes. Hardcoding them in weights
means the model cannot adapt to distribution shift without retraining. A tool
can be updated with new statistics. More importantly, models are bad at
calibrating their own output frequencies -- this is exactly the kind of
meta-cognition that external systems do better.

**Caveat**: The tool is only useful if the model (a) decides to call it, and
(b) changes its prediction based on the result. Our experiments show the
fine-tuned model ignores text corrections entirely (0% scene change rate in
conditioned prediction experiments). So the tool is useful *in principle* but
requires a model that actually responds to its output.

### 1.4 Confusion Patterns Between Classes

**Category: Statistical (tool)**

From the calibration analysis:

- 50.9% of nominal scenes are misclassified as incident_zone (3,430 errors)
- 34.4% of flagger scenes are misclassified as incident_zone (227 errors)
- 28.0% of flooded scenes are misclassified as incident_zone (173 errors)
- 50.7% of mounted_police scenes are misclassified as nominal (138 errors)
- 41.0% of incident_zone scenes are misclassified as nominal (132 errors)

**Why tool**: These are dataset-level statistics about systematic model failures.
A model cannot introspect on its own confusion matrix -- it does not know which
classes it tends to mix up. A `check_confusion_risk("incident_zone")` tool
returning "this class is predicted 12.5x more than it occurs, and is most often
confused with nominal" gives the model (or a verifier model) a reason to
second-guess itself.

This is the strongest case for a tool in this task. The confusion pattern is
systematic, quantifiable, and not available from the image alone. It is
meta-knowledge about model behavior, which is exactly what tools are designed
to provide.

### 1.5 Visual Similarity Between Classes

**Category: Perceptual (weights)**

Some class pairs are more visually similar than others:

- **incident_zone vs. nominal (with triggers)**: The most confusable pair.
  Both may contain cones, barriers, work zones. Discriminating features are
  contextual (active response vs. passive infrastructure).
- **flagger vs. incident_zone**: Both involve construction/work zones. The
  distinguishing feature is a human with a sign/flag.
- **mounted_police vs. nominal**: A horse on a road is distinctive, but at
  distance or in certain angles, it may look like other roadside objects.
- **flooded vs. nominal**: Wet road vs. standing water is a continuum. Light
  flooding is visually subtle.

**Why weights**: Visual similarity is a property of the images themselves. No
tool can quantify "how similar does this specific image look to class X." This
must be learned from data.

### 1.6 Scene-Specific Context Cues

**Category: Perceptual (weights)**

Beyond primary features, there are contextual cues:

- **Time of day**: Incidents are more visible at night (flashing lights).
  Construction flaggers typically work during daylight.
- **Road type**: Flooding is more common on underpasses and low-lying roads.
  Mounted police are more common in urban/suburban settings.
- **Vehicle density**: Incident zones often show stopped/queued traffic.
  Nominal scenes with triggers may have normal traffic flow.
- **Road markings and signs**: Construction zone signs, detour signs, etc.

**Why weights**: These are visual features extracted from the image. They are
part of the perceptual pipeline. A tool could *describe* these associations
(e.g., "incident zones typically have stopped traffic"), but the model still
needs to perceive whether traffic is stopped.

### 1.7 Summary Table: Scene Classification

| Knowledge Component | Category | Justification |
|---|---|---|
| Visual appearance of each scene | Perceptual (weights) | Must look at the image |
| nominal_triggers vs. real incidents | Perceptual (weights) | Contextual visual judgment |
| Base rates per class | Statistical (tool) | Static facts, bad to memorize |
| Confusion patterns | Statistical (tool) | Meta-knowledge about model behavior |
| Visual similarity between classes | Perceptual (weights) | Property of the images |
| Contextual scene cues | Perceptual (weights) | Visual features from the image |

**Bottom line**: 4 of 6 components are perceptual. The 2 statistical components
(base rates and confusion patterns) are genuine tool candidates -- but they only
help if the model can act on the information.

---

## Section 2: Action Prediction

Action prediction outputs a longitudinal action (null, stop, slowdown, proceed)
and a lateral action (null, lc_left, lc_right).

### 2.1 Scene-Action Compatibility Rules

**Category: Statistical (tool) -- but actually a post-processing rule**

The co-occurrence matrix is perfectly structured:

| Scene | Allowed Longitudinal | Allowed Lateral |
|---|---|---|
| nominal | null only | null only |
| mounted_police | null only | null only |
| incident_zone | stop, slowdown, proceed | null, lc_left, lc_right |
| flooded | stop, slowdown, proceed | null only |
| flagger | stop, slowdown, proceed | null only |

These rules are absolute in the training data. There are zero exceptions:
- nominal never has a non-null action
- Only incident_zone ever has a lane change
- mounted_police never has a non-null action

**Why tool**: This is a deterministic lookup table. Given a scene prediction,
the set of valid actions is fully determined. The model should not have to
memorize these constraints -- they are structural rules, not perceptual
judgments.

**But actually**: This is better described as a post-processing rule than a
tool. If the model predicts scene=nominal and action=stop, you do not need to
*ask* the model to reconsider. You can just override: scene=nominal implies
action=(null, null). No model inference required. Calling a tool, waiting for
the model to process the result, and hoping it changes its answer is strictly
worse than just applying the rule directly.

### 2.2 Recognizing the Need for Longitudinal Response

**Category: Perceptual (weights)**

The model must perceive:
- **stop**: Imminent obstacle, red light, stop sign, flagman holding STOP sign,
  completely blocked road.
- **slowdown**: Approaching hazard, standing water, congestion ahead, work zone
  with reduced speed.
- **proceed**: Hazard is passable, flagman signaling SLOW/proceed, road ahead
  is clearing.
- **null**: No special action needed, normal driving.

**Why weights**: Whether to stop or slow down is determined by what the model
sees in the image. Is there standing water? How deep? Is the flagger showing
STOP or SLOW? Is the incident blocking the road or just the shoulder? These
are perceptual judgments.

### 2.3 Recognizing the Need for Lateral Response

**Category: Perceptual (weights)**

Lane changes are predicted only in incident zones. The model must perceive:
- Is the incident blocking the current lane?
- Is there an adjacent lane available?
- Which direction (left or right) is the available lane?

**Why weights**: Lane availability and obstruction are visual properties of the
scene. The model must see whether the road ahead is blocked and which direction
to move.

### 2.4 Action Severity Calibration

**Category: Hybrid (weights + tool)**

Within a given scene type, the distribution of actions matters:

- **flooded**: slowdown (56%), stop (27%), proceed (17%). Most flooded scenes
  call for slowdown, not stop.
- **flagger**: stop (58%), slowdown (31%), proceed (12%). Most flagger scenes
  call for stop.
- **incident_zone**: slowdown (53% across all lateral), stop (25%), proceed (15%).

The *which action* decision is perceptual (what does the image show?), but the
*calibration* of how often each action should appear could be informed by
statistical priors.

**Why hybrid**: The model needs to see the image to decide whether to stop or
slow down. But if it systematically over-predicts "stop" for flooded scenes
when "slowdown" is more common, a statistical prior could help recalibrate.
In practice, action prediction accuracy is secondary to scene accuracy in this
task (actions are correct by construction when scene is correct, because
scene-action rules are deterministic for nominal and mounted_police, which
are 81.3% of data).

### 2.5 HLA Thresholds (Waypoint-Derived Actions)

**Category: Statistical (tool)**

High-level actions are derived from waypoints using physical thresholds:
- stop_speed = 0.3 m/s
- hard_decel = -1.5 m/s^2
- slow_decel = -0.5 m/s^2
- proceed_accel = 0.5 m/s^2
- lane_width = 0.2 m (lateral displacement)

**Why tool**: These are engineering constants. The model should not memorize
them. But more importantly, these thresholds are used to *derive* the action
labels from waypoints, not to *predict* actions. In the prediction pipeline,
the model directly outputs action tokens, not waypoints-then-derive. So these
thresholds are relevant to evaluation, not to the model's inference.

### 2.6 Summary Table: Action Prediction

| Knowledge Component | Category | Justification |
|---|---|---|
| Scene-action compatibility | Statistical (tool) / post-processing | Deterministic rules |
| Need for longitudinal response | Perceptual (weights) | Visual judgment |
| Need for lateral response | Perceptual (weights) | Visual judgment |
| Action severity calibration | Hybrid | Perception + statistical prior |
| HLA physical thresholds | Statistical (tool) | Engineering constants |

**Bottom line**: The single most valuable piece of information for action
prediction -- scene-action compatibility -- is better implemented as a
post-processing rule than a tool. The model does not need to "learn" or "look
up" that nominal scenes have null actions; the system can just enforce it.

---

## Section 3: Waypoint Prediction

Waypoints are 10 (x, y) delta pairs on a 63x63 grid, representing the planned
trajectory.

### 3.1 Road Geometry and Lane Structure

**Category: Perceptual (weights)**

The model must perceive:
- Road curvature (straight, gentle curve, sharp turn)
- Lane markings and boundaries
- Road width
- Intersection geometry
- Whether the vehicle is centered in its lane

**Why weights**: Road geometry is a visual property of the image. The model
needs to see the road to plan a trajectory on it.

### 3.2 Typical Waypoint Ranges per Scene-Action

**Category: Statistical (tool)**

Different scene-action combinations produce different waypoint distributions:

- **nominal + null**: Small deltas centered near (0, 0). The vehicle continues
  straight with minor corrections.
- **flooded + slowdown**: Forward motion decelerating. Y-deltas become
  increasingly negative (deceleration), X-deltas near zero.
- **incident_zone + lc_left**: Lateral shift to the left. X-deltas are
  negative (left) while Y-deltas show deceleration then acceleration.
- **flagger + stop**: Rapid deceleration to zero. Y-deltas become strongly
  negative, converging to zero.

**Why tool**: The typical ranges (mean, std, min, max) per scene-action
combination are statistical summaries. A tool that says "for incident_zone +
lc_left, typical first waypoint X is -0.3 to -0.1" provides a sanity check
on the model's output. However, the waypoint_stats in the current tool_stats.json
are empty (`{}`), meaning we have not yet computed these statistics. This
limits the current utility of the waypoint tool.

### 3.3 Physical Feasibility Constraints

**Category: Statistical (tool) / post-processing**

Waypoints must satisfy physical constraints:
- Consecutive deltas should not imply teleportation (sudden large jumps)
- Acceleration/deceleration should be within physical limits
- Lateral displacement should be within lane-change geometry
- The trajectory should not go off-road

**Why tool/post-processing**: These are kinematic constraints that can be
checked computationally. A tool can verify whether a predicted trajectory is
physically plausible. But again, this is better implemented as post-processing
validation than as something the model consults mid-prediction.

### 3.4 Obstacle Avoidance Paths

**Category: Perceptual (weights)**

When the road is partially blocked (incident, flood, construction), the model
must plan a path around the obstruction. This requires perceiving:
- Where the obstruction is
- How much of the road it blocks
- Where the clear path is

**Why weights**: This is spatial reasoning about the specific image. No lookup
table can tell the model where the obstruction is in this particular scene.

### 3.5 Summary Table: Waypoint Prediction

| Knowledge Component | Category | Justification |
|---|---|---|
| Road geometry and lane structure | Perceptual (weights) | Visual spatial reasoning |
| Typical waypoint ranges per scene-action | Statistical (tool) | Distribution summaries |
| Physical feasibility constraints | Statistical (tool) / post-processing | Kinematic checks |
| Obstacle avoidance paths | Perceptual (weights) | Spatial reasoning about the image |

**Bottom line**: Waypoint prediction is dominated by perceptual requirements.
The statistical components (typical ranges, feasibility checks) are useful for
validation but do not help the model *generate* better waypoints. They can
catch outliers after the fact, which is post-processing.

---

## Section 4: The Tool Design Implications

### 4.1 Which Tools Are Genuinely Useful?

Based on the categorization above, tools are genuinely useful for:

1. **Base rate information** (check_scene_prior): Tells the model how rare
   its prediction is. Genuinely not available from the image. Genuinely hard
   to memorize correctly. Changes if deployment distribution shifts.

2. **Confusion pattern awareness** (check_confusion_risk): Tells the model
   about its own systematic failures. This is meta-cognitive information that
   a model cannot derive from a single image.

These two provide information that is (a) not available from the image, (b) not
trivially memorizable, and (c) actionable if the model responds to it.

### 4.2 Which "Tools" Are Actually Post-Processing Rules?

Three of the knowledge components we identified are better implemented as
deterministic post-processing:

1. **Scene-action compatibility**: If scene=nominal, force action=(null, null).
   No model inference needed. This is a hard constraint, not a suggestion.
   Framing it as a tool that the model "consults" adds latency and
   unreliability (the model might ignore the tool's advice) for no benefit
   over just applying the rule.

2. **Physical feasibility of waypoints**: Check whether waypoints satisfy
   kinematic constraints. If they violate constraints, clamp them. No model
   inference needed for this check.

3. **Action override based on scene**: If the scene prediction changes
   (e.g., incident_zone -> nominal), the action should automatically change
   to match (any non-null action -> null). This is a cascading rule, not a
   tool consultation.

**The key distinction**: A tool is useful when the model needs to *reason*
about the information to decide what to do. A post-processing rule is
sufficient when the correct action is deterministic given the inputs. All
three cases above are deterministic.

### 4.3 Where Is the Model the Bottleneck?

The model is the bottleneck for all perceptual components, which are the
majority of the task:

- **Distinguishing nominal_triggers from real incidents**: This is the single
  failure that causes 85.6% of all errors. No tool can fix it. The model sees
  a cone and predicts incident_zone. A tool can tell the model "incident_zone
  is rare, are you sure?" but the model is sure -- it sees a cone. The
  perception is not wrong (there IS a cone), the interpretation is wrong
  (a cone does not necessarily mean an incident).

- **Visual scene recognition in general**: The model must correctly identify
  what it sees. If it cannot tell a horse from a barrier, no amount of
  statistical context helps.

- **Road geometry for waypoints**: The model must see the road to plan a
  trajectory. No tool provides this.

### 4.4 The Optimal Split

| Responsibility | Owner | Mechanism |
|---|---|---|
| Image perception | Model (weights) | Forward pass through vision encoder |
| Scene classification | Model (weights) | Learned from training data |
| Action type selection | Model (weights) + post-processing | Model proposes, rules enforce |
| Waypoint generation | Model (weights) | Spatial reasoning from image |
| Prior calibration | Tool | `check_scene_prior` |
| Confusion awareness | Tool | `check_confusion_risk` |
| Scene-action enforcement | Post-processing rule | Deterministic override |
| Waypoint feasibility | Post-processing rule | Kinematic clamp |
| Class-conditional action | Post-processing rule | Cascade from scene correction |

The optimal split puts **perception in the model**, **meta-cognition in tools**,
and **deterministic constraints in post-processing**. Most of what matters is
perception.

---

## Section 5: Mapping to Our 4 Tools

### 5.1 check_scene_prior

**What it does**: Returns the base rate of the predicted scene class and flags
whether it is rare (<5%).

**Is it genuinely useful?**

In theory, yes. Incident_zone has a 3.7% base rate, and the model predicts it
at 46.8%. A tool that flags "you are predicting a 3.7% class 46.8% of the time"
is delivering genuinely useful information.

In practice, the utility depends entirely on the verifier model. The fine-tuned
model ignores this information completely (0% scene change rate in conditioned
prediction experiments). A base model verifier *might* respond to it, but only
if it can reconcile the statistical signal with what it sees in the image (which
it may not even have access to in the verify-then-revise pipeline, or may not
interpret correctly as a base model).

**Could this be baked into the prompt instead?** Yes, trivially. The prompt
could say: "Note: incident_zone occurs in only 3.7% of real driving scenes.
If you are predicting incident_zone, double-check that you see active emergency
response, not just construction cones." This is equivalent to the tool's output
but requires no function call overhead.

**Verdict**: The tool form adds no value over a well-written prompt *for this
specific task* because the base rates are static and small enough to enumerate.
In a task with 100+ classes or dynamic distributions, the tool form would be
preferable. Here, it is pedagogically useful for teaching tool calling, but
practically equivalent to a system prompt instruction.

### 5.2 check_scene_action_compatibility

**What it does**: Checks whether a (scene, long_action, lat_action) triple has
been observed in the training data. Returns the observation count and typical
actions for that scene.

**Is it genuinely useful?**

The information is useful: if the model predicts (nominal, stop, null), the tool
says "this combination has never been observed." That is a clear signal.

But this is a **deterministic rule, not a probabilistic signal**. The
compatibility constraints are absolute:
- nominal -> (null, null) always
- mounted_police -> (null, null) always
- Only incident_zone allows lane changes

A rule that enforces these constraints is strictly better than a tool that
*suggests* them to a model that may or may not listen. The tool adds latency
and the possibility of the model ignoring correct advice.

**Could this be baked into the prompt?** Yes, and more effectively as a
post-processing rule. The prompt could state the constraints, but
post-processing can enforce them without relying on model compliance.

**Verdict**: This tool is solving the wrong problem in the wrong way. The
knowledge it provides is deterministic and should be enforced, not suggested.
As a learning exercise, it illustrates an important lesson: not everything that
looks like a tool call is best served by a tool call. Sometimes the right
answer is a constraint, not a consultation.

### 5.3 check_waypoint_feasibility

**What it does**: Checks whether the first waypoint delta is within the typical
statistical range (z-score < 3) for the given scene-action combination.

**Is it genuinely useful?**

Currently, no. The waypoint_stats in tool_stats.json are empty (`{}`), so the
tool has no data to check against. Even with data, this tool can only catch
extreme outliers (z > 3). It cannot improve waypoint quality -- only flag
implausible ones.

Even if fully populated, this tool addresses the *validation* problem, not the
*generation* problem. It is a post-hoc check, not a generative aid. And it only
checks the first waypoint, missing trajectory-level issues (e.g., a trajectory
that starts reasonable but curves off-road).

**Could this be post-processing instead?** Yes. A waypoint feasibility check
does not need model involvement at all. Compute the z-scores, clamp outliers,
done.

**Verdict**: This is the weakest of the four tools. It provides minimal
information, addresses a secondary concern (waypoints vs. scene classification),
and is better implemented as post-processing. Its pedagogical value is in
showing that not all tool calls are created equal.

### 5.4 check_confusion_risk

**What it does**: Returns whether the predicted class is commonly confused with
another class, the error rate, and a descriptive note with remediation guidance.

**Is it genuinely useful?**

This is the strongest tool of the four. It provides:
1. **The fact** that incident_zone is confused with nominal at a 50.9% rate
2. **The direction** of confusion (which class it is confused with)
3. **Actionable guidance** ("Look carefully: are there actual emergency
   vehicles, crashes, or road closures? If not, this is likely nominal.")

This is meta-cognitive information that no single image can provide. A model
looking at a dashcam image cannot know that "models like me tend to predict
incident_zone when they see cones in nominal scenes." This tool provides that
knowledge.

**Could this be baked into the prompt?** Partially. The prompt could say
"be careful about incident_zone over-prediction." But the tool provides
quantified specifics (50.9% error rate, 12.5x over-prediction) that make
the warning more compelling. More importantly, the tool format allows the
model to query this information *conditionally* -- only when it predicts a
potentially confusable class -- rather than overloading the system prompt
with warnings for all classes.

**Verdict**: Genuinely useful as a tool. The conditional nature (only consulted
for specific predictions) and the quantified specificity make the tool format
superior to a static prompt for this information. However, its effectiveness
still depends on the verifier model actually acting on the information.

### 5.5 Cross-Tool Summary

| Tool | Genuine Tool? | Better as Prompt? | Better as Post-Processing? | Pedagogical Value |
|---|---|---|---|---|
| check_scene_prior | Marginal | Yes (for 5 classes) | No | Medium -- teaches base rate awareness |
| check_scene_action_compatibility | No | No | Yes (deterministic rule) | High -- teaches tool vs. rule distinction |
| check_waypoint_feasibility | No | No | Yes (statistical check) | Medium -- teaches tool quality matters |
| check_confusion_risk | Yes | Partially | No | High -- best example of genuine tool utility |

---

## Section 6: Honest Assessment

### 6.1 The Core Problem

Let us be precise about what is happening:

1. The fine-tuned 2B model has a **perceptual miscalibration**: it interprets
   visual triggers (cones, barriers) as sufficient evidence for incident_zone,
   when the training data says they are not.

2. This miscalibration causes 3,430 errors on nominal scenes (50.9% of all
   nominal samples) and accounts for 85.6% of all errors.

3. The miscalibration is **baked into the weights**. The conditioned prediction
   experiment proved this: when told "the correct scene is nominal," the model
   still outputs incident_zone 100% of the time for affected samples. The scene
   token is perception-locked.

4. The calibration analysis shows that a simple post-processing rule -- "flip
   all incident_zone predictions to nominal" -- achieves 84.8% accuracy (up from
   46.9%), saves 3,430 samples, and breaks only 163. A slightly smarter rule
   using confusion-awareness achieves 86.7% accuracy with 0 breaks.

### 6.2 Is Tool Calling the Right Paradigm for This Task?

**No.** Here is why.

**The accuracy ceiling of post-processing is already high.** Strategy 3
(confusion-aware flipping) from the calibration analysis achieves:
- 86.7% accuracy (vs. 46.9% baseline)
- 0.610 macro F1 (vs. 0.429 baseline)
- 3,430 saves, 0 breaks

This requires zero model inference, zero tool calls, and zero latency. It is a
deterministic rule applied after the fine-tuned model's greedy prediction.

**The oracle ceiling shows what tool calling COULD achieve.** Strategy 4 (oracle
with perfect fine-class knowledge) achieves:
- 92.4% accuracy
- 0.775 macro F1
- 3,918 saves, 0 breaks

The gap between confusion-aware post-processing (0.610 F1) and the oracle
ceiling (0.775 F1) is 0.165 F1 points. This gap represents the value of
*perfectly* distinguishing nominal_triggers from real incidents -- a perceptual
task that no tool can perform.

**Tool calling sits in an awkward middle.** For tools to improve over
post-processing, the verifier model must:
1. Actually call the tools (not guaranteed with small models)
2. Correctly interpret tool outputs (requires reasoning capability)
3. Override the original prediction based on tool output (requires the model
   to trust the tool over its own perception)
4. Make the RIGHT override (not just any change)

Each step has a failure probability. The compound probability of all four
succeeding is substantially less than 1. Post-processing has a compound
probability of exactly 1 (it is deterministic).

**The fundamental mismatch**: Tool calling is designed for situations where the
model needs *external information* to make a decision. But the dominant error
in this task is *perceptual*, not informational. The model does not lack
information -- it lacks correct visual interpretation. Telling a model "this
class is rare" does not help if the model is confident it sees an incident.
This is like giving a colorblind person a chart of color frequencies -- the
issue is not that they do not know red is common, it is that they cannot
distinguish red from green.

### 6.3 What Tool Calling Teaches Us

Despite being the wrong paradigm for *maximizing F1 on this specific task*, the
tool-calling experiments are valuable as a **learning laboratory**. Here is what
they teach:

**Lesson 1: The Weights-vs-Tools Decomposition is Real and Important.**
Not everything a model needs to know should be in its weights. Base rates,
confusion patterns, and compatibility rules are genuinely better stored
externally. But recognizing WHICH knowledge belongs where requires careful
analysis -- the default assumption should be "weights" unless there is a clear
reason for "tool."

**Lesson 2: Tools are Not Magic -- They Require a Competent Consumer.**
The fine-tuned model ignores tool-like information entirely (0% scene change
rate). Even a base model verifier may struggle with small model sizes. The
tool is only as good as the model's ability to act on its output. This is the
most underappreciated failure mode of tool calling: the tool works perfectly,
but the model does not listen.

**Lesson 3: Post-Processing Rules are Underrated.**
When constraints are deterministic, post-processing beats tool calling on every
dimension: accuracy (100% compliance vs. probabilistic), latency (zero vs.
multi-turn inference), and simplicity (a few lines of code vs. a tool
framework). The scene-action compatibility rules in this task are a perfect
example. The industry fixation on tool calling sometimes obscures simpler,
better solutions.

**Lesson 4: The Value of Tool Calling Scales with Task Complexity.**
In this task, there are 5 scene classes and 15 scene-action combinations.
All constraints can be enumerated in a short prompt. Tool calling adds overhead
without proportional benefit. In a task with hundreds of classes, dynamic
constraints, or real-time data feeds (e.g., checking current traffic
conditions, weather APIs, live incident databases), tools become essential
because the information cannot fit in a prompt and changes over time.

**Lesson 5: Perceptual Failures Cannot Be Fixed Downstream.**
The most important lesson. If the model's perception is wrong, no amount of
post-processing, tool calling, or prompt engineering fixes it. The 2,801
nominal_trigger errors exist because the model's visual encoder maps those
images to the wrong region of feature space. The only fixes are:
- Better training data (more nominal_trigger examples with correct labels)
- Better visual features (larger model, better pretraining)
- A second model that CAN distinguish the cases (ensembling)

Tool calling gives us none of these. It gives us statistical context, which
is the wrong medicine for a perceptual disease.

### 6.4 The Bottom Line

| Approach | Best Accuracy | Macro F1 | Saves/Breaks | Latency | Complexity |
|---|---|---|---|---|---|
| Baseline (fine-tuned greedy) | 46.9% | 0.429 | -- | 1x | None |
| Post-processing (IZ -> nominal) | 84.8% | 0.474 | 3430/163 | 1x | Trivial |
| Confusion-aware PP | 86.7% | 0.610 | 3430/0 | 1x | Low |
| Oracle ceiling | 92.4% | 0.775 | 3918/0 | 1x | Oracle |
| Tool calling (realistic) | TBD | TBD | TBD | 2-3x | High |

For maximizing this task's F1: use confusion-aware post-processing. It is fast,
deterministic, and already achieves 71% of the oracle ceiling's F1 improvement
with zero additional model inference.

For learning about tool calling as a paradigm: run the experiments. The gap
between "tool provides correct information" and "model acts on correct
information" is the most important thing to understand about tool-augmented
LLMs, and this task provides a clean, measurable testbed for studying it.

The honest answer is that both are true simultaneously. The task is not ideal
for tool calling, but it is excellent for *studying* tool calling, precisely
because the failure modes are so clear and measurable.
