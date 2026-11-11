"""
Simplified AI System - Chatbot, Calculator, Code Checker, Fake News Detector
For Google Colab with ngrok tunneling

============================================================================
COLAB INSTALLATION COMMAND (Copy and run this in a Colab cell first):
============================================================================

!pip install flask pyngrok transformers torch sympy google-genai tree-sitter-language-pack sentence-transformers chromadb openai langchain langchain-community python-dotenv tiktoken accelerate ipywidgets

# Optional: For better performance with GPU
!pip install bitsandbytes

# After installation, set your API keys in Colab Secrets:
# - GEMINI_API_KEY (for chatbot)
# - NGROK_AUTH_TOKEN (for public URL access)

============================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================
from typing import List, Dict, Optional
import json
import re
import os
import math
import sympy
from sympy import sympify, solve, simplify, diff, integrate, limit
from sympy.parsing.sympy_parser import parse_expr

import numpy as np
import difflib
import threading
import ast
import compileall

import os

import importlib
for m in ["transformers", "torch", "accelerate", "bitsandbytes"]:
    try:
        mod = importlib.import_module(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "MISSING:", e)

import torch
print("GPU available:", torch.cuda.is_available())

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai not installed. Install with: pip install google-genai")

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: flask not installed. Install with: pip install flask")

try:
    from pyngrok import ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    print("Warning: pyngrok not installed. Install with: pip install pyngrok")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")


# ============================================================================
# ADVANCED CALCULATOR
# ============================================================================

class AdvancedCalculator:
    """Advanced calculator for complex math and code evaluation."""

    def __init__(self):
        self.history = []

    def calculate(self, expression: str) -> Dict:
        """Calculate a mathematical expression."""
        try:
            expr = expression.replace('^', '**').replace('√', 'sqrt')
            result = eval(expr, {"__builtins__": None}, {
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'sqrt': math.sqrt, 'abs': abs,
                'round': round, 'e': math.e, 'E': math.e, 'pi': math.pi
            })
            self.history.append({'expression': expression, 'result': result})
            return {'success': True, 'result': str(result), 'method': 'basic'}
        except:
            pass

        try:
            expr = parse_expr(expression, evaluate=False)
            result = simplify(expr)
            self.history.append({'expression': expression, 'result': str(result)})
            return {'success': True, 'result': str(result), 'method': 'symbolic'}
        except Exception as e:
            return {'success': False, 'error': f'Could not evaluate expression: {str(e)}'}

    def solve_equation(self, equation: str) -> Dict:
        """Solve algebraic equations."""
        try:
            if '=' in equation:
                left, right = equation.split('=')
                left_expr = parse_expr(left)
                right_expr = parse_expr(right)
                equation_expr = left_expr - right_expr
            else:
                equation_expr = parse_expr(equation)

            variables = list(equation_expr.free_symbols)
            if not variables:
                return {'success': False, 'error': 'No variable found in equation'}

            if len(variables) == 1:
                x = variables[0]
                solutions = solve(equation_expr, x)
                result = [str(sol) for sol in solutions]
                self.history.append({'equation': equation, 'solutions': result})
                return {'success': True, 'solutions': result}
            else:
                return {'success': False, 'error': 'Multiple variables not supported yet'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def derivative(self, expression: str, variable: str = 'x') -> Dict:
        """Calculate derivative."""
        try:
            expr = parse_expr(expression)
            var_symbol = parse_expr(variable)
            result = diff(expr, var_symbol)
            self.history.append({'expression': expression, 'variable': variable, 'derivative': str(result)})
            return {'success': True, 'result': str(result)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def integral(self, expression: str, variable: str = 'x') -> Dict:
        """Calculate indefinite integral."""
        try:
            expr = parse_expr(expression)
            var_symbol = parse_expr(variable)
            result = integrate(expr, var_symbol)
            self.history.append({'expression': expression, 'variable': variable, 'integral': str(result)})
            return {'success': True, 'result': str(result)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_history(self) -> List[Dict]:
        """Get calculation history."""
        return self.history


# ============================================================================
# CODE CHECKER WITH HUGGING FACE
# ============================================================================
from typing import Dict, List

# ============================================================================
# CODE CHECKER  --  REPLACE your whole old `class CodeChecker:` with this one
# (everything between the "CODE CHECKER WITH HUGGING FACE" banner and the
#  "FAKE NEWS DETECTOR" banner).
#
# What changed:
#   * Model: Qwen2.5-Coder-Instruct (chat model made for code review/repair)
#     instead of codegen-350M / santacoder (completion models that can't
#     follow instructions).
#   * ONE model call returns: what is wrong + why, the fixed code, and why
#     the fixed code is correct (the old version made 3 separate calls).
#   * The fix is VERIFIED: it is re-parsed, and if it still has syntax errors
#     the model is shown the error and tries again.
#   * Deterministic decoding (no random sampling) so fixes are stable.
#   * Better offline fallback when no model can load.
#
# Needs: transformers, torch, accelerate. For the 7B model on a Colab GPU
# also:  !pip install bitsandbytes
# Optional: set env CODE_MODEL="some/hf-model" to force a different model.
# ============================================================================
class CodeChecker:

    LANGUAGE_ALIASES = {
        "python": "python",
        "py": "python",
        "javascript": "javascript",
        "js": "javascript",
        "java": "java",
        "c++": "cpp",
        "cpp": "cpp"
    }

    # (model name, load in 4-bit?)  -- tried in order, first one that loads wins.
    GPU_MODELS = [
        ("Qwen/Qwen2.5-Coder-7B-Instruct", True),    # best, ~5 GB VRAM in 4-bit
        ("Qwen/Qwen2.5-Coder-3B-Instruct", False),   # ~7 GB in fp16
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", False),
    ]
    CPU_MODELS = [
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", False),
        ("Qwen/Qwen2.5-Coder-0.5B-Instruct", False),
    ]

    MAX_CHARS = 12000  # longer code is not sent to the model

    SYSTEM_PROMPT = (
        "You are a meticulous senior software engineer doing a code review. "
        "You find real bugs, explain them clearly to the author, and return a "
        "corrected version of the code. You never invent problems."
    )

    OUTPUT_FORMAT = """Reply in EXACTLY this format and nothing else:

<verdict>has_problems or correct</verdict>
<problems>
- Line <n>: <what is wrong> -- <why it is wrong: the concrete situation or input that makes it fail and what the program does wrong>. Example: <a short snippet showing the buggy behavior or the wrong way vs the right way>
</problems>
<fixed_code>
<the complete corrected program, no markdown fences>
</fixed_code>
<why_correct>
- <for each change: what you changed, why the new version is correct, and which edge cases it now handles (empty input, zero, None, etc.)>
</why_correct>
<other_ways>
- <a different correct way to write this program or fix this problem> -- <when it is better than the fixed version, with its trade-off>. Example: <1-3 lines of code showing that approach>
</other_ways>

Rules:
- Look for: syntax errors, logic errors, wrong results, unhandled edge cases (empty input, zero, None/null, index out of range), wrong types, security problems, resource leaks.
- Explain EVERY problem thoroughly: what is wrong, the exact input/situation that triggers the failure, and what goes wrong because of it. Do not just name the bug.
- EVERY problem bullet must include an "Example:" part showing the wrong vs right way (keep it to 1-3 short lines).
- Fix ONLY real problems. Keep the author's names, structure and style.
- fixed_code must be complete and runnable, never a fragment.
- If the code is already correct: verdict = correct, problems = "- No problems found.", repeat the code unchanged in fixed_code, and use why_correct to explain what the code does right and which edge cases it already handles.
- other_ways: if a genuinely different correct approach exists, list up to 2 of them, each with a code example and a note on when it is better. Write "- None." if there is only one sensible way.
- Do not invent problems."""

    IMPROVE_FORMAT = """Reply in EXACTLY this format and nothing else:

<verdict>has_tips or clean</verdict>
<analysis>
- <2-5 sentences: what is suboptimal, risky or inefficient in this code, the concrete situation where it hurts (large input, many calls, untrusted data, etc.), and the real-world impact>
</analysis>
<tips>
- Line <n>: <tip> -- <full explanation of why the current code is weak, how the tip makes it better, and what could go wrong without it>. Before: <the current weak way, 1 line>. Example: <the improved way, 1-3 short lines of code>
</tips>
<other_ways>
- <a genuinely different correct approach> -- <why it can be better and its trade-off>. Example: <1-3 lines of code showing that approach>
</other_ways>
<improved_code>
<the complete improved program, no markdown fences>
</improved_code>
<why_better>
- <why the improved version is better than the original: concrete gains like speed, memory, readability, safety; mention measurable differences where possible>
</why_better>

Rules:
- analysis: explain in full sentences what is weak, when it actually causes problems, and the real-world impact. Do not be vague.
- EVERY tip must have a full explanation, a "Before:" line showing the current weak code, and an "Example:" with the better code. Do not be vague.
- other_ways: up to 3 genuinely different correct approaches, each with a code example and its trade-offs. Write "- None." if only one sensible way exists.
- improved_code: the complete improved program, keeping the author's intent, names and structure. Apply the best of your suggestions. Only real improvements - no invented features, no bugs.
- why_better: 2-4 bullets explaining the concrete gains (faster, safer, shorter, easier to maintain, handles more cases...).
- If the code is already good: verdict = clean, set analysis/tips/other_ways to "- None.", repeat the code unchanged in improved_code, and set why_better to "- Already well written.".
- Line numbers count from the first line of the code shown."""

    STATIC_WHY = {
        "eval()": "eval() runs any text as code, so untrusted input can take control of the program. Example: eval('__import__(\"os\").system(\"rm -rf /\")') would actually run it. Use ast.literal_eval() for data instead.",
        "exec()": "exec() runs dynamically supplied code, which is dangerous with untrusted input. Example: exec(user_text) lets a user run anything on your machine. Prefer explicit functions or ast.literal_eval() for data.",
        "bare 'except:'": "It also swallows KeyboardInterrupt/SystemExit and hides real bugs. Example: pressing Ctrl+C during except: does nothing, and a typo bug gets silently ignored. Catch a specific exception like 'except ValueError:'.",
        "mutable default": "The default object is created once and shared by every call, so data leaks between calls. Example: def add(x, items=[]): items.append(x) keeps old items forever. Use None and create the object inside the function.",
    }

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_name = None
        self._gen_lock = threading.Lock()
        self.analysis_engine = "Python AST + Tree-sitter (no AI model loaded)"
        #threading.Thread(target=self._load_model, daemon=True).start()
        self._load_model()

    # -----------------------------------------
    # MODEL LOADING
    # -----------------------------------------

    def _load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not installed, code fixing will use basic rules only.")
            return
        try:
            import gc
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            print(f"Could not import model libraries: {e}")
            return

        gpu = torch.cuda.is_available()
        try:
            import bitsandbytes  # noqa: F401
            has_bnb = True
        except Exception:
            has_bnb = False

        candidates = []
        forced = os.getenv("CODE_MODEL")
        if forced:
            candidates.append((forced, False))
        candidates += self.GPU_MODELS if gpu else self.CPU_MODELS

        for name, quant in candidates:
            if quant and not (gpu and has_bnb):
                print(f"Skipping {name} (needs a GPU and bitsandbytes for 4-bit loading)")
                continue
            try:
                print(f"Loading code model {name} ...")
                tokenizer = AutoTokenizer.from_pretrained(name)
                kwargs = {}
                if gpu:
                    kwargs["device_map"] = "auto"
                    if quant:
                        from transformers import BitsAndBytesConfig
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                    else:
                        kwargs["torch_dtype"] = torch.float16
                else:
                    kwargs["torch_dtype"] = torch.float32
                model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                model.eval()
                self.tokenizer, self.model, self.model_name = tokenizer, model, name
                self.analysis_engine = f"Python AST + Tree-sitter + {name}"
                print(f"✓ Loaded {name} for code review and fixing")
                return
            except Exception as e:
                print(f"Could not load {name}: {e}")
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
        print("Warning: no code model could be loaded, using basic rules only.")

    def _generate(self, messages: List[Dict], max_new_tokens: int) -> str:
        import torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = {}
        chk = getattr(self, "_cancel_check", None)
        if chk is not None:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList

                class _CancelStop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kw):
                        try:
                            return bool(chk())
                        except Exception:
                            return False

                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_CancelStop()])
            except Exception:
                pass
        with self._gen_lock, torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # deterministic: same code -> same fix
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -----------------------------------------
    # MAIN ENTRY POINT
    # -----------------------------------------

    def check_code(self, code: str, language: str = "python", mode: str = "fix") -> Dict:

        language = self.LANGUAGE_ALIASES.get((language or "").lower().strip())

        if not code or not code.strip():
            return {
                "success": False,
                "language": language or "unknown",
                "syntax_errors": ["No code was provided."],
                "issues": [],
                "recommendations": ["Enter some code to analyze."]
            }

        if language is None:
            return {
                "success": False,
                "language": "unknown",
                "syntax_errors": [
                    "Unsupported language. Supported languages: Python, JavaScript, Java, C++."
                ],
                "issues": [],
                "recommendations": []
            }

        syntax_errors = self._check_syntax(code, language)

        issues = []
        if language == "python" and not syntax_errors:
            issues = self._find_python_issues(code)
        if "TODO" in code or "FIXME" in code:
            issues.append(
                "The code contains TODO or FIXME comments. "
                "Check whether unfinished work remains."
            )

        too_long = len(code) > self.MAX_CHARS
        mode = "improve" if (mode or "").lower() == "improve" else "fix"
        gemini = getattr(self, "_gemini_bot", None)
        improve = None
        fix = None
        if mode == "improve" and not too_long:
            if gemini is not None:
                improve = _cp_gemini_tips(gemini, self, code, language, issues)
            elif self.model:
                improve = self._improve_with_model(code, language, issues)
        if improve is not None:
            imp_code = (improve.get("improved_code") or "").strip()
            changed = bool(imp_code) and imp_code != code.strip()
            status = improve.get("status") or ("improved" if changed else "tips")
            if status == "clean":
                status = "correct"
            recommendations = self._get_recommendations(
                code, language, syntax_errors, issues,
                {"status": "suggestion_only" if syntax_errors else "correct",
                 "problems": [], "fixed_code": code, "why_correct": [], "verified": True},
                too_long)
            return {
                "success": True,
                "language": language,
                "mode": "improve",
                "syntax_errors": syntax_errors,
                "issues": issues,
                "problems": [],
                "analysis": improve.get("analysis") or [],
                "tips": improve.get("tips") or [],
                "other_ways": improve.get("other_ways") or [],
                "why_better": improve.get("why_better") or [],
                "why_correct": [],
                "fixed_code": imp_code if changed else code,
                "status": status,
                "verified": True,
                "changed": changed,
                "kind": "improve",
                "model": self.model_name,
                "recommendations": recommendations,
                "analysis_engine": self.analysis_engine,
            }
        if not too_long:
            if gemini is not None:
                fix = _cp_gemini_review(gemini, self, code, language, syntax_errors, issues)
            if fix is None and self.model:
                fix = self._fix_with_model(code, language, syntax_errors, issues)
        if fix is None:
            fix = self._fix_with_rules(code, language, syntax_errors, issues)

        recommendations = self._get_recommendations(
            code, language, syntax_errors, issues, fix, too_long
        )

        return {
            "success": True,
            "language": language,
            "mode": "fix",
            "syntax_errors": syntax_errors,
            "issues": issues,
            "problems": fix["problems"],          # what is wrong + why
            "fixed_code": fix["fixed_code"],      # the corrected code
            "why_correct": fix["why_correct"],    # why the fix is right
            "other_ways": fix.get("other_ways") or [],  # alternative correct approaches
            "status": fix["status"],
            "verified": fix["verified"],
            "changed": fix["fixed_code"].strip() != code.strip(),
            "auto_fix_available": fix["fixed_code"].strip() != code.strip(),
            "kind": "fix",
            "model": self.model_name,
            "recommendations": recommendations,
            "analysis_engine": self.analysis_engine,
        }

    # -----------------------------------------
    # AI REVIEW + FIX + VERIFY
    # -----------------------------------------

    def _fix_with_model(self, code, language, syntax_errors, issues):
        parts = [f"Language: {language}", "", "Code to review:", f"```{language}", code, "```", ""]
        if syntax_errors:
            parts.append("The parser already reported these syntax errors (they are real):")
            parts += [f"- {e}" for e in syntax_errors]
            parts.append("")
        if issues:
            parts.append("Static analysis also flagged:")
            parts += [f"- {i}" for i in issues]
            parts.append("")
        parts.append(self.OUTPUT_FORMAT)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]
        max_tokens = min(6000, len(code) // 2 + 1500)

        best = None
        for attempt in range(2):
            try:
                reply = self._generate(messages, max_tokens)
            except Exception as e:
                print(f"Code model error: {e}")
                break

            parsed = self._parse_reply(reply, code)
            if parsed is None:
                break

            # The model says it's fine and the parser agrees -> done.
            if parsed["verdict"] == "correct" and not syntax_errors:
                parsed["fixed_code"] = code
                parsed["verified"] = True
                best = parsed
                break

            errors = self._real_syntax_errors(parsed["fixed_code"], language)
            parsed["verified"] = not errors
            best = parsed
            if not errors:
                break

            # Show the model its own mistake and let it try once more.
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    "Your fixed_code still fails to parse:\n"
                    + "\n".join(f"- {e}" for e in errors[:5])
                    + "\nFix it and answer again in the same format."
                )},
            ]

        if best is None:
            return None

        changed = best["fixed_code"].strip() != code.strip()
        if best["verdict"] == "correct" and not syntax_errors:
            status = "correct"
        elif not changed:
            status = "suggestion_only"
        elif best["verified"]:
            status = "fixed"
        else:
            status = "fixed_unverified"

        return {
            "problems": best["problems"],
            "fixed_code": best["fixed_code"],
            "why_correct": best["why_correct"],
            "other_ways": best.get("other_ways") or [],
            "status": status,
            "verified": best["verified"],
        }

    def _improve_with_model(self, code, language, issues):
        """Full improvement review: analysis + tips with examples + other right
        ways + the complete improved program + why it is better."""
        parts = [f"Language: {language}", "", "Code to review:", f"```{language}", code, "```", ""]
        if issues:
            parts.append("Static analysis also flagged:")
            parts += [f"- {i}" for i in issues]
            parts.append("")
        parts.append(self.IMPROVE_FORMAT)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT + " This time improve the code: explain what is weak, give tips with examples, list other right ways, and return the full improved program."},
            {"role": "user", "content": "\n".join(parts)},
        ]
        try:
            reply = self._generate(messages, min(9000, len(code) // 2 + 2000))
        except Exception as e:
            print(f"Improve model error: {e}")
            return None
        verdict = (self._section(reply, "verdict") or "").lower()
        analysis = self._bullets(self._section(reply, "analysis") or "")
        tips = self._bullets(self._section(reply, "tips") or "")
        other = [o for o in self._bullets(self._section(reply, "other_ways") or "")
                 if not re.match(r"(?i)^none\b", o.strip())]
        why = self._bullets(self._section(reply, "why_better") or "")
        imp = self._strip_fences((self._section(reply, "improved_code") or "").strip())
        clean = ("clean" in verdict or
                 (not tips and not analysis and "no improvement" in " ".join(tips).lower()))
        if clean:
            return {"tips": [], "analysis": [], "other_ways": [], "why_better": [],
                    "improved_code": code, "status": "clean"}
        return {"tips": tips, "analysis": analysis, "other_ways": other,
                "why_better": why, "improved_code": imp if imp.strip() else code,
                "status": "improved" if imp.strip() and imp.strip() != code.strip() else "tips"}

    def _parse_reply(self, reply: str, original: str):
        fixed = self._section(reply, "fixed_code")
        if not fixed or not fixed.strip():
            return None
        fixed = self._strip_fences(fixed)

        problems = self._bullets(self._section(reply, "problems") or "")
        why = self._bullets(self._section(reply, "why_correct") or "")
        other = [o for o in self._bullets(self._section(reply, "other_ways") or "")
                 if not re.match(r"(?i)^none\b", o.strip())]

        verdict_raw = (self._section(reply, "verdict") or "").lower()
        if "has_problems" in verdict_raw or "problem" in verdict_raw:
            verdict = "has_problems"
        elif "correct" in verdict_raw:
            verdict = "correct"
        else:
            verdict = "correct" if fixed.strip() == original.strip() else "has_problems"

        return {"verdict": verdict, "problems": problems,
                "fixed_code": fixed, "why_correct": why, "other_ways": other}

    @staticmethod
    def _section(text: str, tag: str):
        m = re.search(rf"<{tag}>[ \t]*\n?(.*?)\n?[ \t]*(?:</{tag}>|$)", text, re.DOTALL)
        return m.group(1) if m else None

    @staticmethod
    def _strip_fences(code: str) -> str:
        code = code.strip("\n")
        m = re.match(r"^```[\w+#-]*\n(.*?)\n?```\s*$", code, re.DOTALL)
        return m.group(1) if m else code

    @staticmethod
    def _bullets(block: str) -> List[str]:
        items = []
        for line in block.splitlines():
            s = line.strip()
            if not s:
                continue
            if s[0] in "-*•":
                items.append(s.lstrip("-*• ").strip())
            elif items:
                items[-1] += "\n" + s
            else:
                items.append(s)
        return items

    def _real_syntax_errors(self, code: str, language: str) -> List[str]:
        errors = self._check_syntax(code, language)
        return [e for e in errors if e.startswith(("Syntax error", "Missing syntax"))]

    # -----------------------------------------
    # FALLBACK WHEN NO MODEL IS LOADED
    # -----------------------------------------

    def _fix_with_rules(self, code, language, syntax_errors, issues):
        problems = list(syntax_errors)
        for issue in issues:
            why = next((w for key, w in self.STATIC_WHY.items() if key in issue), "")
            problems.append(f"{issue} -- {why}" if why else issue)

        fixed = code
        why_correct = []

        if language == "python":
            # Add missing ':' using the exact line the compiler points at.
            for _ in range(20):
                try:
                    compile(fixed, "<code-checker>", "exec")
                    break
                except SyntaxError as e:
                    if e.msg and "expected ':'" in e.msg and e.lineno:
                        lines = fixed.split("\n")
                        lines[e.lineno - 1] = lines[e.lineno - 1].rstrip() + ":"
                        fixed = "\n".join(lines)
                        why_correct.append(
                            f"Line {e.lineno}: added the missing ':' -- Python requires it "
                            "to start the block after if/for/while/def/class/try/with. "
                            "Example: 'if x > 0' alone is a syntax error; 'if x > 0:' starts the block."
                        )
                    else:
                        break

            new = re.sub(r"^(\s*)except\s*:", r"\1except Exception:", fixed, flags=re.M)
            if new != fixed:
                fixed = new
                why_correct.append(
                    "Replaced bare 'except:' with 'except Exception:' -- it no longer "
                    "catches KeyboardInterrupt/SystemExit, so Ctrl+C still works. "
                    "Example: 'except:' also catches SystemExit from sys.exit(); 'except Exception:' does not."
                )

        changed = fixed.strip() != code.strip()
        verified = changed and not self._real_syntax_errors(fixed, language)
        if changed:
            status = "rule_based_fixed" if verified else "fixed_unverified"
        else:
            status = "no_model"
            if not why_correct:
                why_correct = ["The AI model is not loaded, so only basic automatic fixes are available."]

        return {"problems": problems, "fixed_code": fixed,
                "why_correct": why_correct, "other_ways": [], "status": status, "verified": verified}

    # -----------------------------------------
    # SYNTAX CHECKING (unchanged)
    # -----------------------------------------

    def _check_syntax(self, code: str, language: str) -> List[str]:

        errors = []

        if language == "python":
            try:
                compile(code, "<code-checker>", "exec")
            except SyntaxError as error:
                errors.append(
                    f"Syntax error at line {error.lineno}, "
                    f"column {error.offset}: {error.msg}"
                )
            except Exception as error:
                errors.append(str(error))
            return errors

        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(language)
            tree = parser.parse(code.encode("utf-8"))
            self._collect_tree_sitter_errors(tree.root_node, errors)
        except ImportError:
            errors.append(
                "Tree-sitter is not installed. "
                "Run: pip install -U tree-sitter-language-pack"
            )
        except Exception as error:
            errors.append(f"Could not analyze this language: {str(error)}")

        return errors[:10]

    def _collect_tree_sitter_errors(self, node, errors):

        if len(errors) >= 10:
            return
        if node.type == "ERROR" or node.is_missing:
            line = node.start_point.row + 1
            column = node.start_point.column + 1
            if node.is_missing:
                message = f"Missing syntax element at line {line}, column {column}."
            else:
                message = f"Syntax error at line {line}, column {column}."
            errors.append(message)

        # Only descend into subtrees that can contain an error - clean files
        # skip the whole-tree walk, which made big-file scans very slow.
        if node.has_error or node.is_missing:
            for child in node.children:
                if len(errors) >= 10:
                    return
                self._collect_tree_sitter_errors(child, errors)

    # -----------------------------------------
    # PYTHON CODE QUALITY CHECKING (unchanged)
    # -----------------------------------------

    def _find_python_issues(self, code: str) -> List[str]:

        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        for node in ast.walk(tree):

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "eval":
                    issues.append(
                        f"Line {node.lineno}: use of eval() "
                        "can execute untrusted input. "
                        "Avoid it unless strictly necessary."
                    )
                elif node.func.id == "exec":
                    issues.append(
                        f"Line {node.lineno}: use of exec() "
                        "can execute dynamically supplied code. "
                        "Use it with extreme caution."
                    )

            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(
                    f"Line {node.lineno}: bare 'except:' "
                    "catches almost every exception. "
                    "Catch a specific exception instead."
                )

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + [d for d in node.args.kw_defaults if d]:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append(
                            f"Line {node.lineno}: mutable default "
                            "argument detected. Use None instead "
                            "and create the object inside the function."
                        )

        return issues

    # -----------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------

    def _get_recommendations(self, code, language, syntax_errors, issues, fix, too_long) -> List[str]:

        recs = []
        found_problems = bool(syntax_errors or issues) or fix["status"] in (
            "fixed", "fixed_unverified", "suggestion_only", "rule_based_fixed")

        if too_long:
            recs.append(
                f"The code is longer than {self.MAX_CHARS} characters, so the AI review was "
                "skipped. Paste one function or file section at a time."
            )
        if fix["status"] == "fixed_unverified":
            recs.append("The corrected code could not be fully verified. Read it carefully before using it.")
        if fix["status"] in ("fixed", "fixed_unverified", "rule_based_fixed"):
            recs.append(
                "Run the corrected code on your own test inputs. The parser can prove the "
                "syntax is valid, but only testing proves the logic is right."
            )
        if len(code.splitlines()) > 150:
            recs.append("Consider splitting this code into smaller functions or modules.")
        if not found_problems and fix["status"] != "no_model":
            recs.append(
                "No problems were detected. This does not guarantee that the program "
                "works correctly in every case."
            )
        return recs

# ============================================================================
# FAKE NEWS DETECTOR
# ============================================================================

class FakeNewsDetector:
    """Fake news detection using Hugging Face models with website checking."""

    def __init__(self):
        self.classifier = None
        self.trusted_sources = [
            'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'wsj.com', 'bloomberg.com', 'economist.com', 'nature.com',
            'science.org', 'pnas.org', 'cdc.gov', 'who.int', 'nasa.gov'
        ]

        # Detect GPU
        self.device = -1
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
        except:
            pass

        if TRANSFORMERS_AVAILABLE:
            try:
                import torch
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == 0 else torch.float32
                )
                print("✓ Loaded sentiment analysis model for fake news detection")
            except:
                try:
                    import torch
                    self.classifier = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=self.device,
                        torch_dtype=torch.float16 if self.device == 0 else torch.float32
                    )
                    print("✓ Loaded DistilBERT model for fake news detection")
                except:
                    print("Warning: Could not load fake news detection model")

    def detect(self, text: str) -> Dict:
        """Detect if text might be fake/misleading with source checking."""
        sources_found = self._extract_sources(text)
        suspicious_patterns = self._check_suspicious_patterns(text)
        source_analysis = self._analyze_sources(sources_found)

        if TRANSFORMERS_AVAILABLE and self.classifier:
            try:
                sentiment_result = self.classifier(text[:512])
            except:
                sentiment_result = None
        else:
            sentiment_result = None

        credibility_score = self._calculate_credibility(text, suspicious_patterns, source_analysis, sentiment_result)
        verdict = self._get_verdict(credibility_score, source_analysis, suspicious_patterns)

        return {
            'success': True,
            'sources_found': sources_found,
            'source_analysis': source_analysis,
            'suspicious_patterns': suspicious_patterns,
            'sentiment_analysis': sentiment_result,
            'credibility_score': credibility_score,
            'verdict': verdict,
            'recommendation': self._get_recommendation(credibility_score, source_analysis)
        }

    def _extract_sources(self, text: str) -> List[str]:
        """Extract URLs/domains from text."""
        import re
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, text)

        domains = []
        for url in urls:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            if domain:
                domains.append(domain)

        return domains

    def _analyze_sources(self, sources: List[str]) -> Dict:
        """Analyze if sources are trustworthy."""
        trusted_count = 0
        untrusted_count = 0
        no_sources = len(sources) == 0

        for source in sources:
            source_lower = source.lower()
            if any(trusted in source_lower for trusted in self.trusted_sources):
                trusted_count += 1
            else:
                untrusted_count += 1

        return {
            'trusted_count': trusted_count,
            'untrusted_count': untrusted_count,
            'no_sources': no_sources,
            'has_trusted_sources': trusted_count > 0
        }

    def _check_suspicious_patterns(self, text: str) -> List[str]:
        """Check for common fake news patterns."""
        patterns = []
        text_lower = text.lower()

        sensational_words = ['shocking', 'unbelievable', 'you won\'t believe', 'secret',
                           'banned', 'censored', 'mainstream media won\'t tell you',
                           'the truth they\'re hiding', 'exposed', 'miracle', ' BREAKING']
        if any(word in text_lower for word in sensational_words):
            patterns.append("Contains sensationalist language")

        if 'source' not in text_lower and 'according to' not in text_lower and 'reported by' not in text_lower:
            patterns.append("No clear sources cited")

        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            patterns.append("Excessive use of capitalization")

        clickbait = ['click here', 'must read', 'you need to see this', 'this will shock you', 'won\'t believe']
        if any(phrase in text_lower for phrase in clickbait):
            patterns.append("Clickbait-style headline")

        conspiracy = ['cover up', 'conspiracy', 'deep state', 'false flag', 'hoax', 'they don\'t want you to know']
        if any(word in text_lower for word in conspiracy):
            patterns.append("Contains conspiracy theory keywords")

        urgency = ['urgent', 'breaking', 'happening now', 'don\'t ignore', 'immediate action required']
        if any(word in text_lower for word in urgency):
            patterns.append("Creates false sense of urgency")

        return patterns

    def _calculate_credibility(self, text: str, suspicious_patterns: List[str],
                             source_analysis: Dict, sentiment_result) -> float:
        """Calculate a credibility score (0-100)."""
        score = 100.0

        # Penalize suspicious patterns
        score -= len(suspicious_patterns) * 12

        # Source analysis
        if source_analysis['no_sources']:
            score -= 25
        elif source_analysis['has_trusted_sources']:
            score += 20
        elif source_analysis['untrusted_count'] > 0:
            score -= 15

        # Length check
        if len(text) < 100:
            score -= 15

        # Sentiment check (extreme bias reduces credibility)
        if sentiment_result:
            for result in sentiment_result if isinstance(sentiment_result, list) else [sentiment_result]:
                if result.get('score', 0) > 0.95:
                    score -= 10

        return max(0, min(100, score))

    def _get_verdict(self, score: float, source_analysis: Dict, patterns: List[str]) -> str:
        """Get final verdict."""
        if score >= 80:
            return "LIKELY CREDIBLE"
        elif score >= 60:
            return "POSSIBLY CREDIBLE"
        elif score >= 40:
            return "UNVERIFIED"
        else:
            return "LIKELY FAKE"

    def _get_recommendation(self, score: float, source_analysis: Dict) -> str:
        """Get recommendation."""
        if source_analysis['no_sources']:
            return "⚠️ No sources cited. Look for this news on trusted sources like Reuters, AP, BBC, or NYT."
        elif not source_analysis['has_trusted_sources']:
            return f"⚠️ Sources found but not from trusted outlets. Verify on {', '.join(self.trusted_sources[:3])}."
        elif score < 50:
            return "⚠️ This news has multiple red flags. Cross-reference with trusted sources before sharing."
        else:
            return "✓ This appears to be credible, but always verify with multiple trusted sources."


# ============================================================================
# TEXT CORRECTION MODULE
# ============================================================================
class TextCorrector:
    """Advanced text correction for misspellings, dictation errors, and grammar."""

    def __init__(self):
        self.spell_model = None
        self.grammar_model = None
        self.common_misspellings = self._load_common_misspellings()
        self.common_dictation_errors = self._load_dictation_errors()

        if TRANSFORMERS_AVAILABLE:
            try:
                # Load a spell correction model
                from transformers import pipeline
                self.spell_model = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    device=-1
                )
            except:
                try:
                    self.spell_model = pipeline(
                        "text2text-generation",
                        model="facebook/bart-large",
                        device=-1
                    )
                except:
                    print("Warning: Could not load spell correction model")

    def _load_common_misspellings(self) -> Dict[str, str]:
        """Load common misspellings and their corrections."""
        return {
            "teh": "the", "recieve": "receive", "seperate": "separate",
            "definately": "definitely", "occured": "occurred", "untill": "until",
            "wich": "which", "thier": "their", "loose": "lose",
            "begining": "beginning", "sucess": "success", "accomodate": "accommodate",
            "necessery": "necessary", "priviledge": "privilege", "maintainance": "maintenance",
            " recomend": "recommend", "recomend": "recommend", "dont": "don't",
            "cant": "can't", "wont": "won't", "im": "I'm", "your": "you're",
            "its": "it's", "there": "their", "to": "too", "then": "than"
        }

    def _load_dictation_errors(self) -> Dict[str, str]:
        """Load common voice dictation errors."""
        return {
            "write": "right", "right": "write", "two": "to", "too": "to",
            "four": "for", "new": "knew", "know": "no", "no": "know",
            "hear": "here", "here": "hear", "where": "were", "were": "where",
            "buy": "by", "by": "buy", "break": "brake", "brake": "break",
            "peace": "piece", "piece": "peace", "sent": "send", "send": "sent",
            "wait": "weight", "weight": "wait", "week": "weak", "weak": "week"
        }

    def correct_text(self, text: str, aggressive: bool = False) -> Dict:
        """
        Correct text for spelling, dictation errors, and grammar.

        Args:
            text: Input text to correct
            aggressive: Whether to apply more aggressive corrections

        Returns:
            Dict with corrected text and correction details
        """
        if not text or not text.strip():
            return {"original": text, "corrected": text, "corrections": []}

        corrections = []
        corrected = text

        # Step 1: Fix common misspellings
        corrected, spell_fixes = self._fix_misspellings(corrected)
        corrections.extend(spell_fixes)

        # Step 2: Fix common dictation errors
        corrected, dictation_fixes = self._fix_dictation_errors(corrected)
        corrections.extend(dictation_fixes)

        # Step 3: Fix punctuation and spacing
        corrected, punctuation_fixes = self._fix_punctuation(corrected)
        corrections.extend(punctuation_fixes)

        # Step 4: Use ML model for advanced correction if available
        if self.spell_model and aggressive:
            corrected, ml_fixes = self._ml_correction(corrected)
            corrections.extend(ml_fixes)

        # Step 5: Fuzzy matching for severe typos
        corrected, fuzzy_fixes = self._fuzzy_correct(corrected)
        corrections.extend(fuzzy_fixes)

        return {
            "original": text,
            "corrected": corrected,
            "corrections": corrections,
            "changes_made": len(corrections) > 0
        }

    def _fix_misspellings(self, text: str) -> tuple:
        """Fix common misspellings."""
        corrections = []
        words = text.split()
        fixed_words = []

        for word in words:
            lower_word = word.lower().strip(".,!?;:'\"()[]{}")
            if lower_word in self.common_misspellings:
                # Preserve capitalization
                if word[0].isupper():
                    corrected = self.common_misspellings[lower_word].capitalize()
                else:
                    corrected = self.common_misspellings[lower_word]
                fixed_words.append(corrected)
                corrections.append(f"Fixed '{word}' → '{corrected}'")
            else:
                fixed_words.append(word)

        return " ".join(fixed_words), corrections

    def _fix_dictation_errors(self, text: str) -> tuple:
        """Fix common voice dictation errors based on context."""
        corrections = []
        words = text.split()
        fixed_words = []
        prev_word = ""

        for i, word in enumerate(words):
            lower_word = word.lower().strip(".,!?;:'\"()[]{}")

            # Context-aware correction
            if lower_word in self.common_dictation_errors:
                # Simple context checking
                correction = self.common_dictation_errors[lower_word]

                # Preserve capitalization
                if word[0].isupper():
                    correction = correction.capitalize()

                fixed_words.append(correction)
                corrections.append(f"Dictation fix: '{word}' → '{correction}'")
            else:
                fixed_words.append(word)

            prev_word = word

        return " ".join(fixed_words), corrections

    def _fix_punctuation(self, text: str) -> tuple:
        """Fix common punctuation and spacing issues."""
        corrections = []
        corrected = text

        # Fix multiple spaces
        if "  " in corrected:
            corrected = " ".join(corrected.split())
            corrections.append("Fixed multiple spaces")

        # Fix sentence capitalization
        sentences = corrected.split(". ")
        fixed_sentences = []
        for i, sentence in enumerate(sentences):
            if sentence and len(sentence) > 0:
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                    corrections.append(f"Capitalized sentence: '{sentence[:30]}...'")
            fixed_sentences.append(sentence)
        corrected = ". ".join(fixed_sentences)

        # Add missing periods at end
        if corrected and corrected[-1] not in ".!?;:":
            corrected += "."
            corrections.append("Added missing period")

        return corrected, corrections

    def _ml_correction(self, text: str) -> tuple:
        """Use ML model for advanced text correction."""
        corrections = []
        try:
            prompt = f"Fix grammar and spelling: {text}"
            result = self.spell_model(prompt, max_length=200)[0]['generated_text']
            if result and result != text:
                corrections.append("ML-based grammar correction applied")
                return result, corrections
        except Exception as e:
            pass
        return text, corrections

    def _fuzzy_correct(self, text: str) -> tuple:
        """Use fuzzy matching for severe typos."""
        corrections = []
        words = text.split()
        fixed_words = []

        # Build a dictionary of all known correct words
        all_correct = set(self.common_misspellings.values())
        all_correct.update(self.common_dictation_errors.values())
        all_correct.update(self.common_misspellings.keys())
        all_correct.update(self.common_dictation_errors.keys())

        for word in words:
            lower_word = word.lower().strip(".,!?;:'\"()[]{}")
            if len(lower_word) > 3 and lower_word not in all_correct:
                # Try to find close matches
                matches = difflib.get_close_matches(
                    lower_word,
                    all_correct,
                    n=1,
                    cutoff=0.75
                )
                if matches:
                    match = matches[0]
                    if word[0].isupper():
                        match = match.capitalize()
                    fixed_words.append(match)
                    corrections.append(f"Fuzzy fix: '{word}' → '{match}'")
                else:
                    fixed_words.append(word)
            else:
                fixed_words.append(word)

        return " ".join(fixed_words), corrections


# ============================================================================
# CHATBOT
# ============================================================================
class SimpleChatbot:
    """Gemini-powered chatbot with vector memory, a knowledge base, and typo tolerance."""

    CHAT_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    EMBED_MODELS = ["gemini-embedding-001", "text-embedding-004"]

    SYSTEM_PROMPT = (
        "You are pip AI agent, a brilliant, friendly and honest assistant that can answer questions "
        "on any topic: science, math, coding, history, writing, advice, everyday life and more.\n\n"
        "IMPORTANT RULES:\n"
        "- The user's messages often contain spelling mistakes, missing punctuation, broken grammar, "
        "voice-dictation errors, slang or mixed languages. Silently work out what they MEANT and answer "
        "that. Never point out or correct their mistakes unless they ask.\n"
        "- If a message is truly ambiguous, give your best interpretation first, then ask one short "
        "clarifying question.\n"
        "- Reply in the same language the user writes in.\n"
        "- Be clear and accurate. For code, use fenced code blocks. For math, show the steps.\n"
        "- If you are not sure about something, say so instead of guessing.\n"
        "- This app also has tabs for a Calculator, a Code Checker and a Fake News Detector. "
        "Mention them only when they would genuinely help."
    )

    BUILTIN_KNOWLEDGE = [
        "The Calculator tab solves arithmetic, algebraic equations, derivatives and integrals using sympy.",
        "The Code Checker tab checks Python, JavaScript, Java and C++ code for syntax errors and common issues.",
        "The Fake News Detector tab scores how credible a news article is by checking sources, sensational language and suspicious patterns.",
        "The Chat tab is an AI assistant powered by Gemini that can answer questions on any topic.",
        "The pip ai agent logo in the top left opens and closes the sidebar with the four tabs.",
    ]

    def __init__(self, api_key: Optional[str] = None, use_search: bool = True):
        self.history: List[Dict] = []      # [{"role": "user"|"model", "text": str}]
        self.memory: List[Dict] = []       # embedded past turns
        self.knowledge: List[Dict] = []    # embedded knowledge snippets
        self.api_key = api_key
        self.use_search = use_search
        self.client = None
        self.chat_model = os.getenv("GEMINI_MODEL")
        self.embed_model = None
        self.last_error = None
        self._kb_tried = False
        self._discovered = False
        self._lock = threading.Lock()
        self.text_corrector = TextCorrector()  # Initialize text corrector

    # ---------------------------------------------------------------- setup
    def _find_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            return key
    # Read secrets once, in the main thread (Colab blocks this inside Flask threads)
        try:
            from google.colab import userdata
            os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
            print("✓ Gemini key loaded from Colab Secrets")
        except Exception as e:
            print("⚠️ Could not read GEMINI_API_KEY from Colab Secrets:", e)
        return os.environ.get("GEMINI_API_KEY")

    def _ensure_client(self) -> bool:
        if self.client is not None:
            return True
        if not GENAI_AVAILABLE:
            self.last_error = "google-genai is not installed."
            return False
        key = self._find_key()
        if not key:
            self.last_error = "No Gemini API key found. Add GEMINI_API_KEY in Colab Secrets."
            return False
        try:
            self.client = genai.Client(api_key=key)
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    # ----------------------------------------------------------- embeddings
    def _embed(self, texts: List[str]):
        candidates = [self.embed_model] if self.embed_model else self.EMBED_MODELS
        for name in candidates:
            try:
                res = self.client.models.embed_content(model=name, contents=texts)
                vecs = np.array([e.values for e in res.embeddings], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                self.embed_model = name
                return vecs / norms
            except Exception as e:
                self.last_error = str(e)
        return None

    def _ensure_knowledge(self):
        if self._kb_tried:
            return
        self._kb_tried = True
        vecs = self._embed(self.BUILTIN_KNOWLEDGE)
        if vecs is not None:
            for text, vec in zip(self.BUILTIN_KNOWLEDGE, vecs):
                self.knowledge.append({"text": text, "vec": vec})

    def add_knowledge(self, text: str) -> bool:
        """Add your own text so the bot can retrieve it later."""
        if not self._ensure_client():
            return False
        vecs = self._embed([text])
        if vecs is None:
            return False
        self.knowledge.append({"text": text, "vec": vecs[0]})
        return True

    @staticmethod
    def _top_k(qvec, items, k=3, min_score=0.5):
        items = [i for i in items if i.get("vec") is not None]
        if not items:
            return []
        scores = np.vstack([i["vec"] for i in items]) @ qvec
        order = np.argsort(-scores)[:k]
        return [(items[i], float(scores[i])) for i in order if scores[i] >= min_score]

    # ----------------------------------------------------------------- chat
    def chat(self, message: str) -> str:
        with self._lock:
            # Apply text correction to handle misspellings, dictation errors, grammar
            correction_result = self.text_corrector.correct_text(message, aggressive=False)
            corrected_message = message

            # If corrections were made, we'll use the corrected version for processing
            # but keep the original for context if needed
            if self._ensure_client():
                return self._chat_with_gemini(corrected_message)
            return self._simple_response(corrected_message)

    def plain(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024,
              system: str = "") -> str:
        """One-off text completion: no history, no memory, no search.
        Returns the raw answer, or '' when Gemini is unavailable."""
        if not self._ensure_client():
            return ""
        cfg = genai_types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        models = [self.chat_model] if getattr(self, "chat_model", None) else list(self.CHAT_MODELS)
        for m in models[:3]:
            try:
                r = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                t = getattr(r, "text", None)
                if t and t.strip():
                    self.chat_model = m
                    return t.strip()
            except Exception as e:
                self.last_error = str(e)
        return ""

    def _build_system_prompt(self, message: str):
        """Adds retrieved knowledge and old conversation to the system prompt."""
        prompt = self.SYSTEM_PROMPT
        qvec = None
        try:
            self._ensure_knowledge()
            vecs = self._embed([message])
            if vecs is not None:
                qvec = vecs[0]
                facts = self._top_k(qvec, self.knowledge, k=2, min_score=0.55)
                older = self._top_k(qvec, self.memory[:-6], k=3, min_score=0.6)
                if facts:
                    prompt += "\n\nRelevant app info:\n" + "\n".join("- " + f["text"] for f, _ in facts)
                if older:
                    prompt += "\n\nRelevant earlier parts of this conversation:\n" + \
                              "\n".join("- " + m["text"] for m, _ in older)
        except Exception as e:
            self.last_error = str(e)
        return prompt, qvec

    def _discover_models(self):
        """Asks Google which models this key can use, so old names never break the bot."""
        chat, embed = [], []
        try:
            for m in self.client.models.list():
                name = (getattr(m, "name", "") or "").replace("models/", "")
                actions = getattr(m, "supported_actions", None) or []
                if "embedContent" in actions or "embedding" in name:
                    embed.append(name)
                elif ("generateContent" in actions and name.startswith("gemini")
                      and "flash" in name
                      and not any(x in name for x in ("image", "tts", "live", "audio", "robotics", "computer"))):
                    chat.append(name)
        except Exception as e:
            self.last_error = f"Could not list models: {e}"
        return chat, embed

    def _try_models(self, models, contents, system_prompt):
        for model_name in models:
            for with_search in ([True, False] if self.use_search else [False]):
                try:
                    config = genai_types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                        max_output_tokens=2048,
                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())] if with_search else None,
                    )
                    response = self.client.models.generate_content(
                        model=model_name, contents=contents, config=config
                    )
                    text = getattr(response, "text", None)
                    if text and text.strip():
                        self.chat_model = model_name
                        return text.strip()
                    self.last_error = f"{model_name}: empty answer."
                except Exception as e:
                    self.last_error = f"{model_name}: {e}"
        return None

    def _chat_with_gemini(self, message: str) -> str:
        system_prompt, _ = self._build_system_prompt(message)

        contents = [
            genai_types.Content(role=h["role"], parts=[genai_types.Part(text=h["text"])])
            for h in self.history[-20:]
        ]
        contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=message)]))

        models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
        reply = self._try_models(models, contents, system_prompt)

        # If every known name failed, ask Google what is available and try those
        if not reply and not self._discovered:
            self._discovered = True
            found_chat, found_embed = self._discover_models()
            if found_embed:
                self.EMBED_MODELS = found_embed[:3]
                self.embed_model = None
            extra = [m for m in found_chat if m not in models][:4]
            if extra:
                reply = self._try_models(extra, contents, system_prompt)

        if not reply:
            return ("Sorry, I couldn't reach Gemini right now.\n"
                    f"Reason: {self.last_error}\n\n"
                    "Check your API key and internet connection, then try again.")

        self.history.append({"role": "user", "text": message})
        self.history.append({"role": "model", "text": reply})
        self._remember(message, reply)
        return reply

    def _remember(self, user_text: str, bot_text: str):
        """Stores the turn as a vector so it can be recalled later."""
        try:
            summary = f"User: {user_text[:400]} | Assistant: {bot_text[:400]}"
            vecs = self._embed([summary])
            if vecs is not None:
                self.memory.append({"text": summary, "vec": vecs[0]})
                self.memory = self.memory[-200:]
        except Exception as e:
            self.last_error = str(e)

    # ------------------------------------------- offline fallback (no key)
    def _simple_response(self, message: str) -> str:
        """Typo-tolerant rule-based replies when Gemini is not available."""
        words = re.findall(r"[a-z']+", message.lower())
        topics = {
            "greeting": ["hello", "hi", "hey", "hola", "greetings"],
            "math": ["calculator", "calculate", "math", "equation", "derivative", "integral", "solve"],
            "code": ["code", "coding", "programming", "python", "javascript", "bug", "syntax"],
            "news": ["news", "fake", "credible", "article", "misinformation"],
            "help": ["help", "features", "capabilities", "can"],
            "thanks": ["thanks", "thank", "thx"],
        }
        found = set()
        for w in words:
            for topic, keys in topics.items():
                if w in keys or difflib.get_close_matches(w, keys, n=1, cutoff=0.75):
                    found.add(topic)

        note = ("\n\n(Gemini is not connected, so I'm in basic mode. "
                f"Reason: {self.last_error})")
        if "greeting" in found:
            return "Hello! I'm pip AI agent. Ask me anything." + note
        if "math" in found:
            return "Use the Calculator tab for equations, derivatives and integrals." + note
        if "code" in found:
            return "Use the Code Checker tab to check your code for errors." + note
        if "news" in found:
            return "Use the Fake News tab to check how credible an article is." + note
        if "thanks" in found:
            return "You're welcome!" + note
        if "help" in found:
            return "I can chat, calculate, check code and detect fake news." + note
        return "I couldn't use the AI model right now, so I can't answer that properly." + note

    # -------------------------------------------------------------- history
    def clear_history(self) -> None:
        """Clear conversation history and vector memory."""
        self.history = []
        self.memory = []

    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.history.copy()

# ============================================================================
# WEB UI
# ============================================================================

class WebUI:
    """Web UI for the simplified AI system."""

    def __init__(self):
        self.calculator = AdvancedCalculator()
        self.code_checker = CodeChecker()
        self.fake_news_detector = FakeNewsDetector()
        self.chatbot = SimpleChatbot()
        self.text_corrector = TextCorrector()
        self.port = self._find_available_port()

        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()

    def _find_available_port(self, start_port: int = 8080, max_attempts: int = 10) -> int:
        """Find an available port."""
        import socket
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return start_port

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Serve the main HTML page."""
            return self.get_html()

        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Chat endpoint."""
            try:
                data = request.json
                message = data.get('message', '')
                if not message or not message.strip():
                    return jsonify({'success': False, 'error': 'Please enter a message.'}), 400

                # SimpleChatbot.chat() already runs the text corrector itself.
                response = self.chatbot.chat(message)
                return jsonify({'success': True, 'response': response})
            except Exception as e:
                print(f"Chat error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/calculate', methods=['POST'])
        def calculate():
            """Calculate endpoint."""
            try:
                data = request.json
                expression = data.get('expression', '')
                operation = data.get('operation', 'calculate')

                if not expression or not expression.strip():
                    return jsonify({'success': False, 'error': 'Please enter an expression.'}), 400

                if operation == 'calculate':
                    result = self.calculator.calculate(expression)
                elif operation == 'solve':
                    result = self.calculator.solve_equation(expression)
                elif operation == 'derivative':
                    variable = data.get('variable', 'x')
                    result = self.calculator.derivative(expression, variable)
                elif operation == 'integral':
                    variable = data.get('variable', 'x')
                    result = self.calculator.integral(expression, variable)
                else:
                    result = {'success': False, 'error': 'Unknown operation'}

                return jsonify(result)
            except Exception as e:
                print(f"Calculator error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/check_code', methods=['POST'])
        def check_code():
            """Code checker endpoint."""
            try:
                data = request.json
                code = data.get('code', '')
                language = data.get('language', 'python')
                mode = data.get('mode', 'fix')

                if not code or not code.strip():
                    return jsonify({'success': False, 'error': 'Please enter some code.'}), 400

                self.code_checker._gemini_bot = _cp_gemini_ok(self)
                result = self.code_checker.check_code(code, language, mode)
                if not result['success']:
                    return jsonify({'success': False, 'error': ' '.join(result['syntax_errors'])}), 400
                return jsonify(result)
            except Exception as e:
                print(f"Code checker error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/detect_fake_news', methods=['POST'])
        def detect_fake_news():
            """Fake news detection endpoint."""
            try:
                data = request.json
                text = data.get('text', '')

                if not text or not text.strip():
                    return jsonify({'success': False, 'error': 'Please enter some text.'}), 400

                result = self.fake_news_detector.detect(text)

                output = f"📊 CREDIBILITY SCORE: {result['credibility_score']}/100\n"
                output += f"🎯 VERDICT: {result['verdict']}\n\n"

                if result['sources_found']:
                    output += "🔗 SOURCES FOUND:\n"
                    for source in result['sources_found']:
                        output += f"  • {source}\n"
                else:
                    output += "⚠️  NO SOURCES CITED\n"

                if result['source_analysis']['has_trusted_sources']:
                    output += f"✓ {result['source_analysis']['trusted_count']} trusted source(s)\n"
                elif result['source_analysis']['untrusted_count'] > 0:
                    output += f"⚠️  {result['source_analysis']['untrusted_count']} unverified source(s)\n"

                if result['suspicious_patterns']:
                    output += "\n🚨 SUSPICIOUS PATTERNS:\n"
                    for pattern in result['suspicious_patterns']:
                        output += f"  • {pattern}\n"
                else:
                    output += "\n✓ No suspicious patterns detected\n"

                output += f"\n💡 {result['recommendation']}"

                result['formatted_output'] = output
                return jsonify(result)
            except Exception as e:
                print(f"Fake news error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/history', methods=['GET'])
        def get_history():
            """Get calculator history."""
            return jsonify({'success': True, 'history': self.calculator.get_history()})
    def get_html(self) -> str:
        """Generate the HTML for the UI."""
        html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>pip AI agent</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=Lato:300,400,700" rel="stylesheet" type="text/css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #000;
            min-height: 100vh;
            color: #e0e0e0;
            overflow-x: hidden;
            overflow-y: auto;
            position: relative;
        }

        /* Stormy Background Canvas */
        #stormy-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
        }

        /* ===================== PARALLAX PIXEL STARS (whole page) ===================== */
        #stars-wrap {
            position: fixed;
            inset: 0;
            overflow: hidden;
            z-index: 1;
            pointer-events: none;
        }
        #stars, #stars2, #stars3 {
            position: absolute;
            top: 0;
            left: 0;
            background: transparent;
        }
        #stars { width: 1px; height: 1px; box-shadow: var(--shadows-small); animation: animStar 50s linear infinite; }
        #stars:after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 1px; height: 1px; background: transparent; box-shadow: var(--shadows-small);
        }
        #stars2 { width: 2px; height: 2px; box-shadow: var(--shadows-medium); animation: animStar 100s linear infinite; }
        #stars2:after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 2px; height: 2px; background: transparent; box-shadow: var(--shadows-medium);
        }
        #stars3 { width: 3px; height: 3px; box-shadow: var(--shadows-big); animation: animStar 150s linear infinite; }
        #stars3:after {
            content: " "; position: absolute; top: 2000px; left: 0;
            width: 3px; height: 3px; background: transparent; box-shadow: var(--shadows-big);
        }
        @keyframes animStar {
            from { transform: translateY(0px); }
            to { transform: translateY(-2000px); }
        }

        /* ===================== PARALLAX PIXEL STARS (inside every panel) ===================== */
        .panel-stars {
            position: absolute;
            inset: 0;
            overflow: hidden;
            border-radius: inherit;
            pointer-events: none;
            z-index: 0;
        }
        .panel-stars .ps { position: absolute; top: 0; left: 0; background: transparent; }
        .panel-stars .ps1 { width: 1px; height: 1px; box-shadow: var(--p-small); animation: pStar 50s linear infinite; }
        .panel-stars .ps1:after {
            content: " "; position: absolute; top: 1600px; left: 0;
            width: 1px; height: 1px; background: transparent; box-shadow: var(--p-small);
        }
        .panel-stars .ps2 { width: 2px; height: 2px; box-shadow: var(--p-medium); animation: pStar 100s linear infinite; }
        .panel-stars .ps2:after {
            content: " "; position: absolute; top: 1600px; left: 0;
            width: 2px; height: 2px; background: transparent; box-shadow: var(--p-medium);
        }
        .panel-stars .ps3 { width: 3px; height: 3px; box-shadow: var(--p-big); animation: pStar 150s linear infinite; }
        .panel-stars .ps3:after {
            content: " "; position: absolute; top: 1600px; left: 0;
            width: 3px; height: 3px; background: transparent; box-shadow: var(--p-big);
        }
        @keyframes pStar {
            from { transform: translateY(0px); }
            to { transform: translateY(-1600px); }
        }

        /* Main container */
        .container {
            position: relative;
            z-index: 10;
            max-width: 1100px;
            margin: 0 auto;
            padding: 100px 20px 80px;
            transition: margin .45s cubic-bezier(.2,.8,.2,1);
        }
        @media (min-width: 900px) {
            body.sidebar-open .container {
                margin-left: 270px;
                margin-right: 20px;
                max-width: none;
            }
        }

        /* ===================== BLOB LOGO BUTTON (top left) ===================== */
        #Button {
            position: fixed;
            top: 0;
            left: 8px;
            z-index: 30;
            height: 127px;
            width: 170px;
            border: none;
            background: none;
            cursor: pointer;
            transition: all ease-out 1s;
        }
        #Button:focus { outline: 2px solid #9ca3cb; }
        #Button:hover { transform: scale(1.05); }
        #Button::-moz-focus-inner { border: 0; }
        #Button svg { outline: none; width: 100%; height: 100%; display: block; }
        .Button__text { fill: #6b7190; }
        .screenreader-only {
            height: 1px;
            overflow: hidden;
            position: absolute;
            width: 1px;
        }
        #bgBlobEnd, #frontBlobEnd, .circle { visibility: hidden; }
        .circle {
            fill: white;
            stroke: white;
            visibility: hidden;
            opacity: 0;
        }
        .frontBlob { fill: #fffff8; }

        /* Rainbow animated border */
        @property --rainbow-angle {
            syntax: '<angle>';
            initial-value: 0deg;
            inherits: false;
        }
        @keyframes rainbow-rotation {
            to { --rainbow-angle: 1turn; }
        }

        /* ===================== LEFT SIDEBAR (tab buttons) ===================== */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            width: 250px;
            z-index: 20;
            padding: 140px 16px 20px;
            background: rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
            border-right: 2px solid rgba(221, 132, 72, 0.5);
            box-shadow: 0 0 25px rgba(221, 132, 72, 0.25);
            transform: translateX(-100%);
            opacity: 0;
            pointer-events: none;
            transition: transform .45s cubic-bezier(.2,.8,.2,1), opacity .3s ease;
            overflow-y: auto;
        }
        .sidebar.open {
            transform: translateX(0);
            opacity: 1;
            pointer-events: auto;
        }
        .tabs { display: flex; flex-direction: column; gap: 14px; }

        /* Particle tab buttons */
        .particle-btn {
            background: rgba(0,0,0,0.8);
            position: relative;
            border: 2px solid rgba(221, 132, 72, 0.5);
            cursor: pointer;
            padding: 15px 24px;
            border-radius: 100px;
            overflow: hidden;
            font-size: 16px;
            width: 100%;
            text-align: left;
            transition: all 0.3s ease;
        }
        .particle-btn:hover {
            background: rgba(221, 132, 72, 0.3);
            border-color: #ffaa5c;
            transform: scale(1.04);
            box-shadow: 0 0 20px rgba(221, 132, 72, 0.6);
        }
        .particle-btn.active {
            background: linear-gradient(135deg, #dd8448, #ffaa5c);
            border-color: #ffaa5c;
            box-shadow: 0 0 30px rgba(221, 132, 72, 0.8);
        }
        .particle-btn span {
            position: relative;
            z-index: 200;
            color: #ffaa5c;
            text-transform: uppercase;
            font-weight: bold;
            transition: color 0.3s ease;
        }
        .particle-btn:hover span { color: #fff; }
        .particle-btn.active span { color: #fff; }
        .particle-btn canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 100;
            pointer-events: none;
        }

        /* Rainbow animated border: tab panels */
        .tab-content {
            display: none;
            padding: 4px;
            border-radius: 24px;
            position: relative;
            isolation: isolate;
            background: transparent;
        }
        .tab-content.active { display: block; animation: panelIn .35s ease; }
        @keyframes panelIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: none; }
        }
        .tab-content::before {
            content: '';
            position: absolute;
            inset: 0;
            padding: 3px;
            border-radius: inherit;
            background: conic-gradient(
                from var(--rainbow-angle),
                hsl(0 100% 62%),
                hsl(45 100% 60%),
                hsl(100 85% 55%),
                hsl(175 90% 55%),
                hsl(225 100% 65%),
                hsl(275 100% 68%),
                hsl(325 100% 65%),
                hsl(360 100% 62%)
            );
            -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            filter: url(#rainbow-glow);
            animation: rainbow-rotation 10s linear infinite;
            pointer-events: none;
            z-index: 0;
        }

        /* Panel background = starry night sky */
        .tab-inner {
            position: relative;
            z-index: 1;
            border: 2px solid transparent;
            border-radius: 24px;
            padding: 30px;
            background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
            backdrop-filter: blur(10px);
        }
        /* everything inside a panel sits above the stars */
        .tab-inner > :not(.panel-stars) {
            position: relative;
            z-index: 1;
        }

        .input-group { margin-bottom: 20px; }
        .input-group label { display: block; margin-bottom: 8px; }
        .field-row { display: grid; grid-template-columns: 1fr 160px; gap: 16px; }
        textarea { min-height: 150px; resize: vertical; font-family: 'Courier New', monospace; }
        #code-input { min-height: 280px; line-height: 1.5; }
        #news-input { min-height: 240px; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.5; }

        .button-group { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }

        /* Electric output box (a bit see-through so the stars show) */
        .output {
            margin-top: 24px;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 20px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            border: 2px solid rgba(221, 132, 72, 0.5);
            position: relative;
        }
        .output:empty { display: none; }
        .output::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border-radius: 20px;
            border: 2px solid #ffaa5c;
            filter: blur(4px);
            pointer-events: none;
        }

        /* Electric inputs */
        input[type="text"], textarea, select {
            width: 100%;
            padding: 15px;
            background: rgba(0,0,0,0.45);
            border: 2px solid rgba(221, 132, 72, 0.5);
            border-radius: 15px;
            color: #e0e0e0;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        input[type="text"]:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #ffaa5c;
            box-shadow: 0 0 15px rgba(221, 132, 72, 0.4);
        }
        select option { background: #111; color: #e0e0e0; }

        h2 {
            color: #ffaa5c;
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(221, 132, 72, 0.5);
        }
        label {
            color: #ffaa5c;
            font-weight: 600;
            text-shadow: 0 0 5px rgba(221, 132, 72, 0.3);
        }

        /* ===================== GENERATE BUTTON (dark blue glow + twinkling stars) ===================== */
        .generate-button {
            --generate-button-star-1-opacity: .25;
            --generate-button-star-1-scale: 1;
            --generate-button-star-2-opacity: 1;
            --generate-button-star-2-scale: 1;
            --generate-button-star-3-opacity: .5;
            --generate-button-star-3-scale: 1;
            --generate-button-dots-opacity: 0;
            --generate-button-shadow-wide: rgba(20, 50, 180, 0.25);

            appearance: none;
            outline: none;
            border: none;
            padding: 16px 24px 16px 20px;
            border-radius: 29px;
            margin: 0;
            background-color: #0a1230;
            color: #8aa0d8;
            min-width: 192px;
            display: flex;
            align-items: center;
            position: relative;
            cursor: pointer;
            -webkit-tap-highlight-color: transparent;
            z-index: 1;
            transform: scale(var(--generate-button-scale, 1)) translateZ(0);
            box-shadow: 0px 0px 120px var(--generate-button-shadow-wide, transparent), 0px 4px 12px rgba(0, 0, 0, 0.05), 0px 1px 2px rgba(0, 0, 0, 0.1), inset 0px 1px 1px var(--generate-button-shadow-inset, rgba(255, 255, 255, 0.06)), 0 0 0 var(--generate-button-shadow-outline, 0px) rgba(30, 70, 220, 0.5);
            transition: transform .3s, background-color .3s, box-shadow .3s, color .3s;
        }
        .generate-button.compact {
            min-width: 0;
            padding: 10px 20px 10px 14px;
            border-radius: 24px;
        }
        .generate-button.compact span { font-size: 14px; line-height: 24px; }
        .generate-button.compact svg.icon { width: 20px; height: 22px; margin-right: 8px; }
        .generate-button:disabled { opacity: .5; cursor: not-allowed; }

        /* dark blue glow under the button */
        .generate-button:before {
            content: '';
            display: block;
            position: absolute;
            right: 20%;
            height: 20px;
            left: 20%;
            bottom: -10px;
            background: #1a3fcc;
            filter: blur(12.5px);
            z-index: 2;
            clip-path: inset(-200% -30% 10px -30% round 29px);
            opacity: 0;
            transition: opacity .4s;
            transform: translateZ(0);
        }
        /* tiny twinkling stars inside the button */
        .generate-button:after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: inherit;
            pointer-events: none;
            z-index: 0;
            background-image:
                radial-gradient(circle at 12% 30%, #ffffff 0, rgba(255,255,255,0) 1.6px),
                radial-gradient(circle at 27% 72%, #cfe0ff 0, rgba(207,224,255,0) 1.4px),
                radial-gradient(circle at 44% 22%, #ffffff 0, rgba(255,255,255,0) 2px),
                radial-gradient(circle at 58% 76%, #ffffff 0, rgba(255,255,255,0) 1.4px),
                radial-gradient(circle at 73% 28%, #cfe0ff 0, rgba(207,224,255,0) 1.8px),
                radial-gradient(circle at 86% 68%, #ffffff 0, rgba(255,255,255,0) 1.6px),
                radial-gradient(circle at 95% 25%, #cfe0ff 0, rgba(207,224,255,0) 1.2px);
            opacity: .35;
            animation: btnTwinkle 2.6s ease-in-out infinite alternate;
        }
        @keyframes btnTwinkle {
            from { opacity: .25; }
            to { opacity: 1; }
        }
        .generate-button span {
            position: relative;
            z-index: 1;
            font-family: 'Poppins', Arial, sans-serif;
            font-weight: 600;
            font-size: 16px;
            line-height: 26px;
            letter-spacing: 0.005em;
            display: block;
            text-align: left;
        }
        .generate-button span:before {
            content: '';
            background-image: linear-gradient(to right, transparent, #0a1230);
            position: absolute;
            inset: 0;
            z-index: 1;
            pointer-events: none;
            opacity: .4;
            transition: opacity .4s;
        }
        .generate-button .stroke { mix-blend-mode: hard-light; }
        .generate-button .stroke svg {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            fill: none;
            stroke-width: .75px;
            stroke: #cfe0ff;
            stroke-dasharray: 1.5 14;
            stroke-dashoffset: 22;
            opacity: 0;
        }
        .generate-button .stroke svg:nth-child(2) {
            stroke-width: 1px;
            stroke-opacity: .5;
            filter: blur(3px);
        }
        .generate-button svg {
            display: block;
            overflow: visible;
            pointer-events: none;
        }
        .generate-button svg.dots {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            z-index: 10;
            fill: #bfd4ff;
            opacity: var(--generate-button-dots-opacity);
        }
        .generate-button svg.icon {
            width: 24px;
            height: 26px;
            margin-right: 12px;
            fill: currentColor;
        }
        .generate-button svg.icon path:nth-child(1) {
            opacity: var(--generate-button-star-1-opacity);
            transform: scale(var(--generate-button-star-1-scale)) translateZ(0);
            transform-origin: 25% 14.58%;
        }
        .generate-button svg.icon path:nth-child(2) {
            opacity: var(--generate-button-star-2-opacity);
            transform: scale(var(--generate-button-star-2-scale)) translateZ(0);
            transform-origin: 60.42% 50%;
        }
        .generate-button svg.icon path:nth-child(3) {
            opacity: var(--generate-button-star-3-opacity);
            transform: scale(var(--generate-button-star-3-scale)) translateZ(0);
            transform-origin: 25% 85.42%;
        }
        .generate-button:hover {
            --generate-button-scale: 1.1;
            --generate-button-shadow-wide: rgba(25, 65, 220, 0.6);
            --generate-button-shadow-inset: rgba(160, 190, 255, 0.35);
            --generate-button-shadow-outline: 3px;
            color: #fff;
            background-color: #0b2a8a;
        }
        .generate-button:hover .stroke svg { animation: stroke 2s linear infinite; }
        .generate-button:hover:before { opacity: 1; }
        .generate-button:hover span:before { opacity: 0; }
        .generate-button:hover:active { --generate-button-scale: 1.05; }
        @keyframes stroke {
            0% { opacity: 0; }
            25%, 75% { opacity: 1; }
            95%, 100% { stroke-dashoffset: 6; opacity: 0; }
        }

        /* ===================== CHAT (ChatGPT / Claude style) ===================== */
        #chat .tab-inner {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            min-height: 480px;
            padding: 20px 24px;
        }
        .chat-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }
        .chat-head h2 { margin-bottom: 0; font-size: 1.3em; }
        .chat-frame {
            position: relative;
            flex: 1;
            min-height: 0;
            border-radius: 20px;
            border: 2px solid rgba(221, 132, 72, 0.5);
            background: rgba(0,0,0,0.2);
        }
        .chat-frame::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            border-radius: 20px;
            border: 2px solid #ffaa5c;
            filter: blur(4px);
            pointer-events: none;
        }
        .chat-container {
            height: 100%;
            overflow-y: auto;
            padding: 24px 22px;
            scroll-behavior: smooth;
            scrollbar-width: thin;
            scrollbar-color: rgba(221,132,72,.6) transparent;
        }
        .chat-container::-webkit-scrollbar { width: 8px; }
        .chat-container::-webkit-scrollbar-thumb { background: rgba(221,132,72,.6); border-radius: 8px; }

        .chat-welcome {
            min-height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 10px;
        }
        .chat-welcome h3 {
            font-size: 2em;
            background: linear-gradient(90deg, #dd8448, #ffaa5c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 20px rgba(221, 132, 72, 0.5);
            margin-bottom: 8px;
        }
        .chat-welcome p { color: #a0a0a0; }
        .chips {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            width: 100%;
            max-width: 560px;
            margin-top: 26px;
        }
        .chips .generate-button { width: 100%; }

        .msg-row { display: flex; gap: 12px; margin-bottom: 22px; }
        .msg-row.user { flex-direction: row-reverse; }
        .avatar {
            width: 34px; height: 34px;
            border-radius: 50%;
            flex: none;
            display: grid;
            place-items: center;
            font-weight: 700;
            font-size: 14px;
            color: #fff;
            background: linear-gradient(135deg, #dd8448, #ffaa5c);
        }
        .msg-row.assistant .avatar {
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(221, 132, 72, 0.6);
            color: #ffaa5c;
        }
        .bubble {
            min-width: 0;
            line-height: 1.65;
            overflow-wrap: anywhere;
        }
        .msg-row.user .bubble {
            background: linear-gradient(135deg, #dd8448, #ffaa5c);
            color: #fff;
            padding: 10px 16px;
            border-radius: 18px 18px 4px 18px;
            max-width: 75%;
            white-space: pre-wrap;
        }
        .msg-row.assistant .bubble { padding-top: 5px; max-width: 85%; }
        .bubble code {
            font-family: 'Courier New', monospace;
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 6px;
            font-size: .92em;
        }
        .bubble pre {
            margin: 10px 0;
            padding: 14px;
            background: rgba(0,0,0,0.7);
            border: 1px solid rgba(221, 132, 72, 0.4);
            border-radius: 12px;
            overflow-x: auto;
        }
        .bubble pre code { background: none; padding: 0; }
        .typing { display: inline-flex; gap: 5px; padding: 12px 0; }
        .typing i {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #ffaa5c;
            opacity: .4;
            animation: blink 1.2s infinite;
        }
        .typing i:nth-child(2) { animation-delay: .2s; }
        .typing i:nth-child(3) { animation-delay: .4s; }
        @keyframes blink { 0%, 80%, 100% { opacity: .3; transform: scale(.8); } 40% { opacity: 1; transform: scale(1.1); } }

        .composer {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            margin-top: 14px;
            padding: 8px 8px 8px 14px;
            border: 2px solid rgba(221, 132, 72, 0.5);
            border-radius: 28px;
            background: rgba(0,0,0,0.3);
            transition: all .3s ease;
        }
        .composer:focus-within { border-color: #ffaa5c; box-shadow: 0 0 15px rgba(221, 132, 72, 0.4); }
        .composer textarea {
            flex: 1;
            min-height: 44px;
            max-height: 180px;
            resize: none;
            border: none;
            background: transparent;
            padding: 10px 4px;
            line-height: 1.5;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .composer textarea:focus { border: none; box-shadow: none; }
        .composer .generate-button { flex: none; }

        @media (max-width: 700px) {
            .field-row { grid-template-columns: 1fr; }
            .chips { grid-template-columns: 1fr; }
            .msg-row.user .bubble { max-width: 88%; }
            .msg-row.assistant .bubble { max-width: 92%; }
            .tab-inner { padding: 20px; }
            .particle-btn { padding: 12px 18px; font-size: 14px; }
        }

        /* ---------- Code checker report ---------- */
        #code-output { white-space: normal; }
        #code-output pre { white-space: pre; }
        .cr-status { padding: 12px 16px; border-radius: 14px; font-weight: 600; margin-bottom: 14px; }
        .cr-status.ok   { background: rgba(60, 200, 120, .15); border: 1px solid rgba(60, 200, 120, .6); color: #7dffb0; }
        .cr-status.warn { background: rgba(255, 170, 92, .15); border: 1px solid rgba(255, 170, 92, .6); color: #ffc58a; }
        .cr-title { color: #ffaa5c; font-weight: 700; margin: 18px 0 8px; display: flex; align-items: center; gap: 10px; }
        #code-output ul { margin: 0; padding-left: 22px; }
        #code-output li { margin-bottom: 8px; line-height: 1.55; }
        #code-output pre {
            margin: 6px 0; padding: 14px; background: rgba(0,0,0,.7);
            border: 1px solid rgba(221, 132, 72, .4); border-radius: 12px; overflow-x: auto;
            font-family: 'Courier New', monospace; font-size: .95em;
        }
        .cr-copy {
            background: rgba(0,0,0,.6); color: #ffaa5c; border: 1px solid rgba(221,132,72,.6);
            border-radius: 10px; padding: 3px 12px; cursor: pointer; font-size: .85em;
        }
        .cr-copy:hover { background: rgba(221,132,72,.3); color: #fff; }
    </style>
</head>
<body>
    <svg aria-hidden="true" focusable="false" width="0" height="0" style="position:absolute; overflow:hidden">
        <filter id="rainbow-glow" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
            <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="soft-glow" />
            <feTurbulence type="fractalNoise" baseFrequency="1.73" numOctaves="2" seed="2" result="border-noise" />
            <feDisplacementMap in="soft-glow" in2="border-noise" scale="14" xChannelSelector="R" yChannelSelector="G" result="textured-glow" />
            <feBlend in="SourceGraphic" in2="textured-glow" mode="screen" />
        </filter>
    </svg>

    <canvas id="stormy-canvas"></canvas>

    <!-- Parallax pixel stars (on top of the storm) -->
    <div id="stars-wrap">
        <div id="stars"></div>
        <div id="stars2"></div>
        <div id="stars3"></div>
    </div>

    <!-- Blob logo (top left) - click to open/close the sidebar -->
    <button id="Button" aria-label="pip ai agent" aria-expanded="false" aria-controls="sidebar">
      <svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" viewBox="0 0 1490.01 1111.6">
        <rect mask="url(#mask)" x="0" y="0" width="100%" height="100%" fill="#c9cfe8" />
        <image mask="url(#mask)" class="Button__background" href="https://image.freepik.com/free-vector/marble-texture-background-pastel-colors_56745-9.jpg" x="0" y="0" height="100%" width="100%" />
        <defs>
          <filter id="gooey" height="150%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="15" result="blur" />
            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
          </filter>
        </defs>
        <mask id="mask">
          <g style="filter: url(#gooey)">
             <path id="bgBlobStart" fill="white" stroke="white" d="M684.62 309.07c77.64-27.55 169.41-56.73 247.52-50.68s142.52 47 152.38 98.64-34.67 114-56.17 166.71-20 95.83-48.4 130.09-86.87 59.79-151.58 97.17c-64.54 37.31-135.3 86.51-193.7 86.17-58.58-.43-104.78-50.3-133.23-94.26s-39.14-82-56.27-119.32-40.66-73.59-36-112.65c4.79-39.19 38-81 86.29-114.87s111.67-59.6 189.16-87z"/>
             <path id="bgBlobEnd" fill="white" stroke="white" d="M905.85 301.31c49.09 16.62 63.94 71.05 116.62 107.69 52.87 36.7 143.73 55.57 162.6 92.43s-34.24 91.8-93.28 136.78-124 80.12-191 99.22c-66.9 19-135.9 22-213.94 33.38s-165.25 31.31-229.68 20.75C392.56 781 350.75 740 336.1 693.43c-14.83-46.55-2.32-98.72 25-148.2s69.63-96.06 126.82-132 129.14-61 207.13-84.18c77.85-23.19 161.54-44.33 210.8-27.74z"/>
             <circle class="circleBottom circle" cx="276.98" cy="628.31" r="23.16"/>
             <circle class="circleBottom circle" cx="528.26" cy="867.81" r="39.52"/>
             <circle class="circleBottom circle" cx="204.98" cy="760.82" r="33.73"/>
             <circle class="circleBottom circle" cx="488.73" cy="954.56" r="23.16"/>
             <circle class="circleBottom circle" cx="371.28" cy="907.33" r="31.26"/>
             <circle class="circleBottom circle" cx="329.79" cy="783.97" r="23.16"/>
             <circle class="circleTop circle" cx="1155.21" cy="286.38" r="33.73"/>
             <circle class="circleTop circle" cx="1044.52" cy="358.74" r="23.16"/>
             <circle class="circleTop circle" cx="965.31" cy="192.59" r="39.52"/>
             <circle class="circleTop circle" cx="1189.14" cy="418.22" r="25.16"/>
             <circle class="circleTop circle" cx="1107.62" cy="136.89" r="33.73"/>
             <circle class="circleTop circle" cx="1295.95" cy="327.49" r="31.26"/>
          </g>
        </mask>
        <path id="frontBlobStart" class="frontBlob" d="M981.72 372.33c56.87 13.9 100.57 39.75 119.1 69.47 18.69 29.61 12.06 63.19 2.24 95.77s-22.64 64.15-59.94 84.86C1006 643.2 944.56 653 882.26 672.51c-62.48 19.52-126 48.76-198 58.09s-152.7-1.24-215.47-25.14-107.61-61.15-106.2-100.18c1.24-38.93 48.72-79.64 103.82-108.11s117.62-44.86 172-66.51 100.38-48.86 158.5-61.47 127.94-10.75 184.81 3.14z"/>
        <path id="frontBlobEnd" class="frontBlob" d="M915.49 434c62.08 31.56 138.58 48.94 170.89 76.61 32.14 27.65 19.92 65.59 13.7 107.53-6.4 41.93-7 87.76-51.66 110.19-44.52 22.45-133.32 21.41-209.14 12.6-75.65-8.7-138.66-25.28-189.62-44.87-51.13-19.6-90.21-42.2-147.44-66.72s-132.41-51.09-166.37-86.25C301.71 508 309 464.35 347 431.66S453.7 377.25 527.43 359c73.56-18.35 151.94-33.34 215.05-17.91S853.24 402.42 915.49 434z"/>
        <text class="Button__text" text-anchor="middle" font-family="'Poppins','Segoe UI',Arial,sans-serif" font-weight="800" font-size="128">
          <tspan x="745" y="525">pip ai</tspan>
          <tspan x="745" y="655">agent</tspan>
        </text>
      </svg>
      <span class="screenreader-only">pip ai agent</span>
    </button>

    <!-- LEFT SIDEBAR: tab buttons -->
    <aside class="sidebar" id="sidebar">
        <div class="tabs">
            <button class="particle-btn active" data-tab="chat">
                <span>💬 Chat</span>
                <canvas></canvas>
            </button>
            <button class="particle-btn" data-tab="calculator">
                <span>🧮 Calculator</span>
                <canvas></canvas>
            </button>
            <button class="particle-btn" data-tab="code">
                <span>💻 Code Checker</span>
                <canvas></canvas>
            </button>
            <button class="particle-btn" data-tab="news">
                <span>📰 Fake News</span>
                <canvas></canvas>
            </button>
        </div>
    </aside>

    <div class="container">

        <!-- CHAT -->
        <div id="chat" class="tab-content active">
            <div class="tab-inner">
                <div class="chat-head">
                    <h2>AI Chatbot</h2>
                    <button class="generate-button compact" onclick="clearChat()"><span>Clear</span></button>
                </div>
                <div class="chat-frame">
                    <div class="chat-container" id="chat-messages">
                        <div class="chat-welcome" id="chat-welcome">
                            <h3>How can I help you today?</h3>
                            <p>Ask me anything, or try one of these:</p>
                            <div class="chips">
                                <button class="generate-button compact" onclick="sendChat('What can you do?')"><span>What can you do?</span></button>
                                <button class="generate-button compact" onclick="sendChat('Help me with math')"><span>Help me with math</span></button>
                                <button class="generate-button compact" onclick="sendChat('Can you check my code?')"><span>Check my code</span></button>
                                <button class="generate-button compact" onclick="sendChat('How do I spot fake news?')"><span>Spot fake news</span></button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="composer">
                    <textarea id="chat-input" rows="1" placeholder="Message pip ai agent..."></textarea>
                    <button class="generate-button compact" id="chat-send" onclick="sendChat()"><span>Send</span></button>
                </div>
            </div>
        </div>

        <!-- CALCULATOR -->
        <div id="calculator" class="tab-content">
            <div class="tab-inner">
                <h2>Advanced Calculator</h2>
                <div class="field-row">
                    <div class="input-group">
                        <label>Expression / Equation</label>
                        <input type="text" id="calc-input" placeholder="e.g., 2+3*4, x^2-4=0, sin(x)">
                    </div>
                    <div class="input-group">
                        <label>Variable</label>
                        <input type="text" id="calc-variable" value="x" placeholder="x">
                    </div>
                </div>
                <div class="button-group">
                    <button class="generate-button compact" onclick="calculate('calculate')"><span>Calculate</span></button>
                    <button class="generate-button compact" onclick="calculate('solve')"><span>Solve Equation</span></button>
                    <button class="generate-button compact" onclick="calculate('derivative')"><span>Derivative</span></button>
                    <button class="generate-button compact" onclick="calculate('integral')"><span>Integral</span></button>
                </div>
                <div class="output" id="calc-output"></div>
            </div>
        </div>

        <!-- CODE CHECKER -->
        <div id="code" class="tab-content">
            <div class="tab-inner">
                <h2>Code Checker</h2>
                <div class="input-group">
                    <label>Programming Language</label>
                    <select id="code-language">
                        <option value="python">Python</option>
                        <option value="javascript">JavaScript</option>
                        <option value="java">Java</option>
                        <option value="cpp">C++</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Your Code</label>
                    <textarea id="code-input" spellcheck="false" placeholder="Paste your code here..."></textarea>
                </div>
                <button class="generate-button" onclick="checkCode()"><span>Check Code</span></button>
                <div class="output" id="code-output"></div>
            </div>
        </div>

        <!-- FAKE NEWS -->
        <div id="news" class="tab-content">
            <div class="tab-inner">
                <h2>Fake News Detector</h2>
                <div class="input-group">
                    <label>News Article Text</label>
                    <textarea id="news-input" placeholder="Paste the news article text here..."></textarea>
                </div>
                <button class="generate-button" onclick="detectFakeNews()"><span>Check Credibility</span></button>
                <div class="output" id="news-output"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/MorphSVGPlugin.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/Physics2DPlugin.min.js"></script>
    <script>
        // ========== PARALLAX PIXEL STARS (whole page) ==========
        (function() {
            const wrap = document.getElementById('stars-wrap');
            const maxX = Math.max(window.innerWidth, 2000);
            function makeShadows(n) {
                const parts = [];
                for (let i = 0; i < n; i++) {
                    parts.push(Math.floor(Math.random() * maxX) + 'px ' + Math.floor(Math.random() * 2000) + 'px #FFF');
                }
                return parts.join(', ');
            }
            wrap.style.setProperty('--shadows-small', makeShadows(700));
            wrap.style.setProperty('--shadows-medium', makeShadows(200));
            wrap.style.setProperty('--shadows-big', makeShadows(100));
        })();

        // ========== PARALLAX PIXEL STARS (inside every panel) ==========
        (function() {
            const root = document.documentElement;
            function makeShadows(n) {
                const parts = [];
                for (let i = 0; i < n; i++) {
                    parts.push(Math.floor(Math.random() * 1400) + 'px ' + Math.floor(Math.random() * 1600) + 'px #FFF');
                }
                return parts.join(', ');
            }
            root.style.setProperty('--p-small', makeShadows(500));
            root.style.setProperty('--p-medium', makeShadows(150));
            root.style.setProperty('--p-big', makeShadows(70));

            document.querySelectorAll('.tab-inner').forEach(inner => {
                const layer = document.createElement('div');
                layer.className = 'panel-stars';
                layer.innerHTML = '<div class="ps ps1"></div><div class="ps ps2"></div><div class="ps ps3"></div>';
                inner.insertBefore(layer, inner.firstChild);
            });
        })();

        // ========== STORMY BACKGROUND ==========
        const stormCanvas = document.getElementById('stormy-canvas');
        const stormCtx = stormCanvas.getContext('2d');
        let frame = 0, brightness = 0;

        function resizeStorm() {
            stormCanvas.width = window.innerWidth;
            stormCanvas.height = window.innerHeight;
        }
        resizeStorm();
        window.addEventListener('resize', resizeStorm);

        function drawBackground() {
            // Sky gradient
            for(let i = 99; i--;) {
                stormCtx.fillStyle = `hsl(0,0%,${99-i}%,1)`;
                stormCtx.fillRect(0, i/199*stormCanvas.height, stormCanvas.width, -stormCanvas.height);
            }

            // Moon
            stormCtx.fillStyle = 'hsla(0,0%,99%,0.5)';
            stormCtx.beginPath();
            stormCtx.ellipse(
                (Math.random()*0.8+0.1)*stormCanvas.width, 99+Math.random()*199,
                99, 99, 0, 0, Math.PI*2
            );
            stormCtx.fill();

            // Mountains
            let g = 9, h = 9;
            for(let j = 9; j--;) {
                for(let i = 0; i < stormCanvas.width; ++i) {
                    stormCtx.fillStyle = `hsl(0,0%,${j*3+g/19}%,1)`;
                    stormCtx.fillRect(i, stormCanvas.height+g-(j/15+0.1)*stormCanvas.height, 1, stormCanvas.height);
                    g += h = Math.random()<0.1 ? (Math.random()*2-1) : h - g/1e4;
                }
            }
        }

        function updateStorm() {
            // Rain
            stormCtx.fillStyle = `hsla(0,0%,${brightness-=2}%,0.4)`;
            stormCtx.fillRect(0, 0, stormCanvas.width, stormCanvas.height);
            stormCtx.rotate(-0.2);
            stormCtx.fillStyle = 'hsla(0,0%,99%,0.2)';
            for(let i = 2000; i--;) {
                stormCtx.fillRect(
                    Math.sin(i**3)*stormCanvas.width+stormCanvas.width/2,
                    (2000+i)/2*(frame/60*(1+Math.sin(i)/4)+9)%(stormCanvas.height+stormCanvas.width/4),
                    Math.sin(i)*2, 50
                );
            }
            stormCtx.rotate(0.2);

            // Lightning
            if(Math.random() < 0.003) {
                brightness = 95;
                stormCtx.fillStyle = 'hsla(0,0%,0%,0.5)';
                let x = (Math.random()*0.8+0.1)*stormCanvas.width;
                stormCtx.beginPath();
                stormCtx.moveTo(x, 0);
                for(let y = 0; y < stormCanvas.height; y += 10) {
                    x += (Math.random()-0.5)*20;
                    stormCtx.lineTo(x, y);
                }
                stormCtx.strokeStyle = '#fff';
                stormCtx.lineWidth = 3;
                stormCtx.stroke();
            }

            frame++;
            requestAnimationFrame(updateStorm);
        }

        drawBackground();
        updateStorm();

        // ========== SIDEBAR TOGGLE (click the pip ai agent blob) ==========
        const sidebar = document.getElementById('sidebar');
        const logoBtn = document.getElementById('Button');

        function setSidebar(open) {
            sidebar.classList.toggle('open', open);
            document.body.classList.toggle('sidebar-open', open);
            logoBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
        }
        logoBtn.addEventListener('click', () => {
            setSidebar(!sidebar.classList.contains('open'));
        });

        // ========== PARTICLE TAB BUTTONS ==========
        const tabButtons = document.querySelectorAll('.particle-btn');
        tabButtons.forEach(btn => {
            const canvas = btn.querySelector('canvas');
            const ctx = canvas.getContext('2d');
            let particles = [];

            function resizeBtnCanvas() {
                canvas.width = btn.offsetWidth;
                canvas.height = btn.offsetHeight;
            }
            resizeBtnCanvas();
            window.addEventListener('resize', resizeBtnCanvas);

            class Particle {
                constructor() {
                    this.x = Math.random() * canvas.width;
                    this.y = Math.random() * canvas.height;
                    this.z = Math.random() * 2;
                    this.vx = (Math.random() - 0.5) * 0.1;
                    this.vy = (Math.random() - 0.5) * 0.1;
                    this.vz = -0.02;
                }
                update() {
                    this.x += this.vx;
                    this.y += this.vy;
                    this.z += this.vz;
                    if(this.z < 0) this.z = 2;
                    if(this.x < 0 || this.x > canvas.width) this.vx *= -1;
                    if(this.y < 0 || this.y > canvas.height) this.vy *= -1;
                }
                render() {
                    const r = ((2 - this.z) / 2) * 2;
                    ctx.beginPath();
                    ctx.fillStyle = 'rgba(255,255,255,0.3)';
                    ctx.arc(this.x, this.y, r, 0, Math.PI*2);
                    ctx.fill();
                }
            }

            for(let i = 0; i < 30; i++) particles.push(new Particle());

            function animateParticles() {
                if(btn.classList.contains('active')) {
                    ctx.fillStyle = 'rgba(0,0,0,0.1)';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    particles.forEach(p => { p.update(); p.render(); });
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
                requestAnimationFrame(animateParticles);
            }
            animateParticles();

            btn.addEventListener('click', () => {
                showTab(btn.dataset.tab);
                // On small screens the sidebar covers the page, so close it after picking a tab
                if (window.innerWidth < 900) setSidebar(false);
            });
        });

        // ========== TAB SWITCHING ==========
        function showTab(tabName) {
            tabButtons.forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
        }

        // ========== GSAP PLUGINS ==========
        if (window.gsap) {
            const plugins = [window.MorphSVGPlugin, window.Physics2DPlugin].filter(Boolean);
            if (plugins.length) gsap.registerPlugin(...plugins);
        }

        // ========== BLOB LOGO ANIMATION ==========
        (function() {
            const logo = document.querySelector('#Button');
            if (!window.gsap || !window.MorphSVGPlugin) return;

            const ease = 'back.out(3)';
            const tl = gsap.timeline({ paused: true });
            tl.set('.circle', { autoAlpha: 0 });
            tl.to('#bgBlobStart', {
                duration: 3,
                morphSVG: { shape: '#bgBlobEnd', type: 'rotational' },
                ease: ease
            }, 0.2)
            .to('#frontBlobStart', {
                duration: 3,
                morphSVG: { shape: '#frontBlobEnd', type: 'rotational' },
                ease: ease
            }, 0)
            .from('.circleTop', { duration: 1.5, x: -100, y: 100, transformOrigin: '50% 50%', ease: ease, stagger: 0.15 }, 0.5)
            .from('.circleBottom', { duration: 1, x: 100, y: -100, transformOrigin: '50% 50%', ease: ease, stagger: 0.15 }, 0.5)
            .to('.circleTop', { duration: 0.1, autoAlpha: 1, ease: ease, stagger: 0.15 }, 0.5)
            .to('.circleBottom', { duration: 0.1, autoAlpha: 1, ease: ease, stagger: 0.15 }, 0.5);

            logo.addEventListener('mouseenter', () => { tl.timeScale(1).play(); });
            logo.addEventListener('mouseleave', () => { tl.timeScale(3).reverse(); });
        })();

        // ========== GENERATE BUTTONS (sparkle + dots) ==========
        const ICON_SVG = '<svg class="icon" viewBox="0 0 24 26">' +
            '<path d="M4 4a2 2 0 1 1 4 0 2 2 0 1 1-4 0"/>' +
            '<path d="M10 13a4.5 4.5 0 1 1 9 0 4.5 4.5 0 1 1-9 0"/>' +
            '<path d="M3 22a3 3 0 1 1 6 0 3 3 0 1 1-6 0"/>' +
            '</svg>';

        const DOT_AMOUNT = 40;

        function createSVG(width, height, className, childType, childAttributes) {
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.classList.add(className);
            const child = document.createElementNS('http://www.w3.org/2000/svg', childType);
            svg.setAttributeNS('http://www.w3.org/2000/svg', 'viewBox', `0 0 ${width} ${height}`);
            for (const attr in childAttributes) child.setAttribute(attr, childAttributes[attr]);
            svg.appendChild(child);
            return { svg, child };
        }

        // Buttons inside hidden tabs have no size until shown, so each button is set up on first hover.
        function setupGenerateButton(button) {
            if (!window.gsap) return null;

            const width = button.offsetWidth;
            const height = button.offsetHeight;
            const radius = parseInt(getComputedStyle(button).borderRadius, 10) || 29;

            const { svg, child: circle } = createSVG(width, height, 'dots', 'circle', { cx: '0', cy: '0', r: '0' });

            const strokeGroup = document.createElement('div');
            strokeGroup.classList.add('stroke');

            const { svg: stroke } = createSVG(width, height, 'stroke-line', 'rect', {
                x: '0', y: '0', width: '100%', height: '100%',
                rx: radius, ry: radius, pathLength: '10'
            });

            button.appendChild(svg);
            strokeGroup.appendChild(stroke);
            strokeGroup.appendChild(stroke.cloneNode(true));
            button.appendChild(strokeGroup);

            const timeline = gsap.timeline({ paused: true });

            for (let i = 0; i < DOT_AMOUNT; i++) {
                const p = circle.cloneNode(true);
                svg.appendChild(p);

                gsap.set(p, {
                    attr: {
                        cx: gsap.utils.random(width * 0.25, width * 0.75),
                        cy: gsap.utils.random(height * 0.5, height * 0.5),
                        r: 0
                    }
                });

                const durationRandom = gsap.utils.random(10, 12);
                const tl = gsap.timeline();

                tl.to(p, {
                    duration: durationRandom,
                    rotation: i % 2 === 0 ? 200 : -200,
                    attr: {
                        r: gsap.utils.random(0.75, 1.5),
                        cy: -width * gsap.utils.random(1.25, 1.75)
                    },
                    physics2D: {
                        angle: -90,
                        gravity: gsap.utils.random(-4, -8),
                        velocity: gsap.utils.random(10, 25)
                    }
                }, '-=' + durationRandom / 2)
                .to(p, { duration: durationRandom / 3, attr: { r: 0 } }, '-=' + durationRandom / 4);

                timeline.add(tl, i / 3);
            }

            svg.removeChild(circle);

            const finalTimeline = gsap.to(timeline, {
                duration: 10,
                repeat: -1,
                time: timeline.duration(),
                paused: true
            });

            const stars = gsap.to(button, {
                repeat: -1,
                repeatDelay: 0.75,
                paused: true,
                keyframes: [
                    {
                        '--generate-button-star-2-scale': '.5',
                        '--generate-button-star-2-opacity': '.25',
                        '--generate-button-star-3-scale': '1.25',
                        '--generate-button-star-3-opacity': '1',
                        duration: 0.3
                    },
                    {
                        '--generate-button-star-1-scale': '1.5',
                        '--generate-button-star-1-opacity': '.5',
                        '--generate-button-star-2-scale': '.5',
                        '--generate-button-star-3-scale': '1',
                        '--generate-button-star-3-opacity': '.5',
                        duration: 0.3
                    },
                    {
                        '--generate-button-star-1-scale': '1',
                        '--generate-button-star-1-opacity': '.25',
                        '--generate-button-star-2-scale': '1.15',
                        '--generate-button-star-2-opacity': '1',
                        duration: 0.3
                    },
                    {
                        '--generate-button-star-2-scale': '1',
                        duration: 0.35
                    }
                ]
            });

            return {
                enter() {
                    gsap.to(button, {
                        '--generate-button-dots-opacity': '.5',
                        duration: 0.25,
                        onStart: () => {
                            finalTimeline.restart().play();
                            setTimeout(() => stars.restart().play(), 500);
                        }
                    });
                },
                leave() {
                    gsap.to(button, {
                        '--generate-button-dots-opacity': '0',
                        '--generate-button-star-1-opacity': '.25',
                        '--generate-button-star-1-scale': '1',
                        '--generate-button-star-2-opacity': '1',
                        '--generate-button-star-2-scale': '1',
                        '--generate-button-star-3-opacity': '.5',
                        '--generate-button-star-3-scale': '1',
                        duration: 0.15,
                        onComplete: () => {
                            finalTimeline.pause();
                            stars.pause();
                        }
                    });
                }
            };
        }

        document.querySelectorAll('.generate-button').forEach(button => {
            button.insertAdjacentHTML('afterbegin', ICON_SVG);
            let api = null;
            button.addEventListener('pointerenter', () => {
                if (!api) api = setupGenerateButton(button);
                if (api) api.enter();
            });
            button.addEventListener('pointerleave', () => { if (api) api.leave(); });
        });

        // ========== CHAT ==========
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const chatSendBtn = document.getElementById('chat-send');
        const chatWelcome = document.getElementById('chat-welcome');
        let chatBusy = false;

        function escapeHtml(s) {
            return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function renderMarkdown(text) {
            return text.split(/(```[\s\S]*?```)/g).map(part => {
                if (part.length >= 6 && part.startsWith('```') && part.endsWith('```')) {
                    const code = part.slice(3, -3).replace(/^[\w+-]*\n/, '').replace(/\n$/, '');
                    return '<pre><code>' + escapeHtml(code) + '</code></pre>';
                }
                return escapeHtml(part)
                    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
                    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
                    .replace(/^- /gm, '• ')
                    .replace(/\n/g, '<br>');
            }).join('');
        }

        function scrollChat() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addRow(role) {
            chatWelcome.style.display = 'none';
            const row = document.createElement('div');
            row.className = 'msg-row ' + role;
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = role === 'user' ? 'U' : 'AI';
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            row.appendChild(avatar);
            row.appendChild(bubble);
            chatMessages.appendChild(row);
            scrollChat();
            return { row, bubble };
        }

        function revealText(bubble, full, done) {
            const step = Math.max(2, Math.ceil(full.length / 120));
            let i = 0;
            const timer = setInterval(() => {
                i = Math.min(full.length, i + step);
                bubble.innerHTML = renderMarkdown(full.slice(0, i));
                scrollChat();
                if (i >= full.length) {
                    clearInterval(timer);
                    done();
                }
            }, 16);
        }

        function autoGrow() {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
        }
        chatInput.addEventListener('input', autoGrow);
        chatInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChat();
            }
        });

        async function sendChat(presetText) {
            if (chatBusy) return;
            const msg = (typeof presetText === 'string' ? presetText : chatInput.value).trim();
            if (!msg) return;

            chatBusy = true;
            chatSendBtn.disabled = true;

            const userRow = addRow('user');
            userRow.bubble.textContent = msg;
            chatInput.value = '';
            autoGrow();

            const bot = addRow('assistant');
            bot.bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
            scrollChat();

            const finish = () => {
                chatBusy = false;
                chatSendBtn.disabled = false;
                chatInput.focus();
            };

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: msg})
                });
                const data = await res.json();
                const reply = data.success ? data.response : 'Error: ' + data.error;
                revealText(bot.bubble, reply, finish);
            } catch (e) {
                bot.bubble.textContent = 'Error: ' + e.message;
                finish();
            }
        }

        function clearChat() {
            chatMessages.querySelectorAll('.msg-row').forEach(r => r.remove());
            chatWelcome.style.display = '';
            chatMessages.scrollTop = 0;
        }

        // ========== CALCULATOR ==========
        document.getElementById('calc-input').addEventListener('keydown', e => {
            if (e.key === 'Enter') calculate('calculate');
        });

        async function calculate(op) {
            const expr = document.getElementById('calc-input').value;
            const variable = document.getElementById('calc-variable').value;
            const out = document.getElementById('calc-output');
            if(!expr || !expr.trim()) {
                out.textContent = 'Please enter an expression.';
                return;
            }
            out.textContent = 'Calculating...';
            try {
                const res = await fetch('/api/calculate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({expression: expr, operation: op, variable: variable})
                });
                const data = await res.json();
                if(data.success) {
                    let output = '';
                    if(data.result) output += 'Result: ' + data.result + '\n';
                    if(data.solutions) output += 'Solutions: ' + data.solutions.join(', ');
                    out.textContent = output;
                } else {
                    out.textContent = data.error;
                }
            } catch(e) {
                out.textContent = 'Error: ' + e.message;
            }
        }
        let lastFixedCode = '';

        function copyFixedCode(btn) {
            navigator.clipboard.writeText(lastFixedCode).then(() => {
                const old = btn.textContent;
                btn.textContent = 'Copied ✓';
                setTimeout(() => { btn.textContent = old; }, 1500);
            });
        }

        function renderCodeReport(d) {
            const list = arr => '<ul>' + arr.map(x => '<li>' + escapeHtml(x) + '</li>').join('') + '</ul>';
            const CODE_CSS = 'display:block;margin-top:5px;padding:7px 9px;background:rgba(127,127,127,.18);border-radius:8px;white-space:pre-wrap;font-size:.92em';
            function bulletHtml(t) {
                t = String(t);
                const marks = [];
                const re = /(Before:|Example:)/g;
                let m;
                while ((m = re.exec(t))) marks.push({ i: m.index, label: m[1] });
                if (!marks.length) return '<li>' + escapeHtml(t) + '</li>';
                let h = '<li>' + escapeHtml(t.slice(0, marks[0].i));
                for (let k = 0; k < marks.length; k++) {
                    const end = k + 1 < marks.length ? marks[k + 1].i : t.length;
                    const code = t.slice(marks[k].i + marks[k].label.length, end).trim();
                    h += '<br><b>' + marks[k].label + '</b><code style="' + CODE_CSS + '">' + escapeHtml(code) + '</code>';
                }
                return h + '</li>';
            }
            const bullets = arr => '<ul>' + arr.map(bulletHtml).join('') + '</ul>';
            const statuses = {
                correct:          ['ok',   '✅ Your code looks correct. Nothing needed fixing.'],
                fixed:            ['ok',   '🔧 Problems found and fixed. The corrected code parses cleanly.'],
                rule_based_fixed: ['ok',   '🔧 Basic problems fixed (AI model not loaded, so only simple rules were used).'],
                fixed_unverified: ['warn', '⚠️ Problems found. This is the best fix, but it could not be fully verified. Read it carefully.'],
                suggestion_only:  ['warn', '⚠️ Problems found, but no safe automatic fix was produced. See the explanations below.'],
                no_model:         ['warn', '⚠️ The AI model is not loaded, so only basic checks were run.']
            };
            const isImprove = d.kind === 'improve' || d.mode === 'improve';
            if (isImprove) {
                statuses.improved = ['ok', '✨ Code improved. Full explanations and the improved program are below.'];
                statuses.tips = ['ok', '💡 Improvement tips (the AI did not return a full rewrite).'];
                statuses.correct = ['ok', '✅ Your code already looks good. No improvements needed.'];
            }
            const st = statuses[d.status] || ['warn', ''];
            let h = st[1] ? '<div class="cr-status ' + st[0] + '">' + st[1] + '</div>' : '';
            if (d.syntax_errors && d.syntax_errors.length) {
                h += '<div class="cr-title">⚠️ Compiler / parser report</div>' + list(d.syntax_errors);
            }
            if (d.analysis && d.analysis.length) {
                h += '<div class="cr-title">🔎 Analysis — what is weak and why it matters</div>' + list(d.analysis);
            }
            if (d.status !== 'correct' && d.problems && d.problems.length) {
                h += '<div class="cr-title">❌ What is wrong, why it fails, and the right way</div>' + bullets(d.problems);
            }
            if (d.tips && d.tips.length) {
                h += '<div class="cr-title">💡 Improvement tips — why it is better, before vs after</div>' + bullets(d.tips);
            }
            if (d.other_ways && d.other_ways.length) {
                h += '<div class="cr-title">🔀 Other right ways to do it</div>' + bullets(d.other_ways);
            }
            if (d.changed) {
                lastFixedCode = d.fixed_code;
                h += '<div class="cr-title">' + (isImprove ? '✨ Improved code' : '✅ Corrected code') +
                     ' <button class="cr-copy" onclick="copyFixedCode(this)">Copy</button></div>' +
                     '<pre><code>' + escapeHtml(d.fixed_code) + '</code></pre>';
            }
            if (d.why_better && d.why_better.length) {
                h += '<div class="cr-title">💡 Why the improved code is better</div>' + list(d.why_better);
            }
            if (d.why_correct && d.why_correct.length) {
                h += '<div class="cr-title">💡 Why this is correct</div>' + bullets(d.why_correct);
            }
            if (d.recommendations && d.recommendations.length) {
                h += '<div class="cr-title">📝 Recommendations</div>' + list(d.recommendations);
            }
            return h;
        }

        async function checkCode() {
            const code = document.getElementById('code-input').value;
            const lang = document.getElementById('code-language').value;
            const out = document.getElementById('code-output');
            if (!code || !code.trim()) {
                out.textContent = 'Please enter some code.';
                return;
            }
            const mode = (document.querySelector('input[name="cu-mode"]:checked') || {}).value || 'fix';
            out.textContent = (mode === 'improve' ? 'Improving code... (the AI review can take up to a minute)' : 'Analyzing code... (the AI review can take up to a minute)');
            try {
                const res = await fetch('/api/check_code', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({code: code, language: lang, mode: mode})
                });
                const data = await res.json();
                if (data.success) {
                    out.innerHTML = renderCodeReport(data);
                } else {
                    out.textContent = data.error || 'Something went wrong.';
                }
                out.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            } catch (e) {
                out.textContent = 'Error: ' + e.message;
            }
        }

        // ========== FAKE NEWS ==========
        async function detectFakeNews() {
            const text = document.getElementById('news-input').value;
            const out = document.getElementById('news-output');
            if(!text || !text.trim()) {
                out.textContent = 'Please enter some text.';
                return;
            }
            out.textContent = 'Analyzing...';
            try {
                const res = await fetch('/api/detect_fake_news', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });
                const data = await res.json();
                if(data.success) {
                    out.textContent = data.formatted_output || 'Analysis complete';
                } else {
                    out.textContent = data.error;
                }
                out.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            } catch(e) {
                out.textContent = 'Error: ' + e.message;
            }
        }
    </script>
</body>
</html>"""

        return html_content

    def launch(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Launch the Flask web server."""
        if not FLASK_AVAILABLE:
            print("⚠️  Flask not installed. Install with: pip install flask")
            return

        if not self.app:
            print("⚠️  Flask app not initialized")
            return

        self.app.run(host=host, port=port, debug=debug)

import tempfile


class VoiceTranscriber:
    """Speech-to-text with OpenAI Whisper from Hugging Face (many languages).
    Loads on the first voice message, so startup stays fast."""

    GPU_MODELS = ["openai/whisper-large-v3-turbo", "openai/whisper-small"]
    CPU_MODELS = ["openai/whisper-small", "openai/whisper-base"]

    def __init__(self):
        self.asr = None
        self.model_name = None
        self._lock = threading.Lock()

    def _load(self):
        if self.asr is not None:
            return True
        if not TRANSFORMERS_AVAILABLE:
            return False
        import torch
        gpu = torch.cuda.is_available()
        for name in (self.GPU_MODELS if gpu else self.CPU_MODELS):
            try:
                print(f"Loading speech model {name} ...")
                self.asr = pipeline(
                    "automatic-speech-recognition",
                    model=name,
                    device=0 if gpu else -1,
                    torch_dtype=torch.float16 if gpu else torch.float32,
                    chunk_length_s=30,
                )
                self.model_name = name
                print(f"✓ Loaded {name} for voice messages")
                return True
            except Exception as e:
                print(f"Could not load {name}: {e}")
        return False

    def transcribe(self, path: str) -> str:
        with self._lock:
            if not self._load():
                raise RuntimeError("Speech model could not be loaded (needs transformers + torch, and ffmpeg).")
            out = self.asr(path, return_timestamps=False, generate_kwargs={"task": "transcribe"})
        return (out.get("text") or "").strip()


def _pip_add_voice_route(self):
    @self.app.route('/api/voice', methods=['POST'])
    def voice():
        f = request.files.get('audio')
        if not f:
            return jsonify({'success': False, 'error': 'No audio received.'}), 400
        mt = (f.mimetype or '').lower()
        suffix = '.ogg' if 'ogg' in mt else '.m4a' if ('mp4' in mt or 'aac' in mt) else '.wav' if 'wav' in mt else '.webm'
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp)
                path = tmp.name
            text = self.voice.transcribe(path)
            if not text:
                return jsonify({'success': False, 'error': "I couldn't hear anything. Try again, a little closer to the mic."}), 400
            reply = self.chatbot.chat(text)
            return jsonify({'success': True, 'transcript': text, 'response': reply})
        except Exception as e:
            print(f"Voice error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


PIP_HEAD = ('<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@500;700'
            '&family=Patrick+Hand&display=swap" rel="stylesheet">')

PIP_CSS = r"""
/* ===== Colors toggle (based on uiverse.io "sun/moon" switch by RiccardoRapelli, MIT license, (c) 2025 RiccardoRapelli) ===== */
.theme-wrap{position:fixed;top:14px;right:22px;z-index:60;display:flex;flex-direction:column;align-items:center;gap:4px;--scale:1.15}
.theme-label{font-family:'Caveat','Segoe UI',cursive;font-weight:700;font-size:22px;line-height:1;color:#fff;text-shadow:0 2px 6px rgba(0,0,0,.5)}
.tsw{position:relative;display:inline-block;width:calc(var(--scale)*60px);height:calc(var(--scale)*34px)}
.tsw #themeInput{opacity:0;width:0;height:0}
.tsw .slider{position:absolute;cursor:pointer;inset:0;background-color:rgba(33,150,243,.5);transition:.4s;z-index:0;overflow:hidden;
  backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.5);
  box-shadow:inset 0 3px 5px rgba(255,255,255,.6),inset 0 -6px 9px rgba(0,0,60,.35),0 10px 18px rgba(0,0,0,.35),0 2px 0 rgba(255,255,255,.3)}
.tsw .slider.round{border-radius:calc(var(--scale)*34px)}
.tsw #themeInput:checked + .slider{background-color:rgba(0,0,0,.55)}
.tsw #themeInput:focus-visible + .slider{outline:2px solid #9ec5ff}
.tsw .sun-moon{position:absolute;height:calc(var(--scale)*26px);width:calc(var(--scale)*26px);left:calc(var(--scale)*4px);bottom:calc(var(--scale)*4px);
  background:radial-gradient(circle at 35% 30%,#fffde7,#ffee58 45%,#fb8c00);transition:.4s;border-radius:50%;
  box-shadow:inset -3px -4px 6px rgba(170,70,0,.5),0 4px 8px rgba(0,0,0,.4)}
.tsw #themeInput:checked + .slider .sun-moon{transform:translateX(calc(var(--scale)*26px));
  background:radial-gradient(circle at 35% 30%,#fff,#e8eaf6 50%,#9fa8da);animation:rotate-center .6s ease-in-out both;
  box-shadow:inset -3px -4px 6px rgba(40,50,120,.45),0 4px 8px rgba(0,0,0,.45)}
@keyframes rotate-center{from{transform:translateX(calc(var(--scale)*26px)) rotate(0)}to{transform:translateX(calc(var(--scale)*26px)) rotate(360deg)}}
.tsw .moon-dot{opacity:0;transition:.4s;fill:gray}
.tsw #themeInput:checked + .slider .sun-moon .moon-dot{opacity:1}
.tsw #moon-dot-1{left:calc(var(--scale)*10px);top:calc(var(--scale)*3px);position:absolute;width:calc(var(--scale)*6px);height:calc(var(--scale)*6px);z-index:4}
.tsw #moon-dot-2{left:calc(var(--scale)*2px);top:calc(var(--scale)*10px);position:absolute;width:calc(var(--scale)*10px);height:calc(var(--scale)*10px);z-index:4}
.tsw #moon-dot-3{left:calc(var(--scale)*16px);top:calc(var(--scale)*18px);position:absolute;width:calc(var(--scale)*3px);height:calc(var(--scale)*3px);z-index:4}
.tsw #light-ray-1,.tsw #light-ray-2,.tsw #light-ray-3{position:absolute;z-index:-1;fill:#fff;opacity:14%}
.tsw #light-ray-1{left:calc(var(--scale)*-8px);top:calc(var(--scale)*-8px);width:calc(var(--scale)*43px);height:calc(var(--scale)*43px)}
.tsw #light-ray-2{left:-50%;top:-50%;width:calc(var(--scale)*55px);height:calc(var(--scale)*55px)}
.tsw #light-ray-3{left:calc(var(--scale)*-18px);top:calc(var(--scale)*-18px);width:calc(var(--scale)*60px);height:calc(var(--scale)*60px)}
.tsw .cloud-light,.tsw .cloud-dark{position:absolute;animation:tcloud-move 6s infinite;filter:drop-shadow(0 3px 2px rgba(0,0,60,.35))}
.tsw .cloud-light{fill:#f4f4f4}.tsw .cloud-dark{fill:#d6d6d6;animation-delay:1s}
.tsw #cloud-1{left:calc(var(--scale)*30px);top:calc(var(--scale)*15px);width:calc(var(--scale)*40px)}
.tsw #cloud-2{left:calc(var(--scale)*44px);top:calc(var(--scale)*10px);width:calc(var(--scale)*20px)}
.tsw #cloud-3{left:calc(var(--scale)*18px);top:calc(var(--scale)*24px);width:calc(var(--scale)*30px)}
.tsw #cloud-4{left:calc(var(--scale)*36px);top:calc(var(--scale)*18px);width:calc(var(--scale)*40px)}
.tsw #cloud-5{left:calc(var(--scale)*48px);top:calc(var(--scale)*14px);width:calc(var(--scale)*20px)}
.tsw #cloud-6{left:calc(var(--scale)*22px);top:calc(var(--scale)*26px);width:calc(var(--scale)*30px)}
@keyframes tcloud-move{0%{transform:translateX(0)}40%{transform:translateX(calc(var(--scale)*4px))}80%{transform:translateX(calc(var(--scale)*-4px))}100%{transform:translateX(0)}}
.tsw .stars{transform:translateY(calc(var(--scale)*-32px));opacity:0;transition:.4s}
.tsw .star{fill:#fff;position:absolute;transition:.4s;animation:tstar-twinkle 2s infinite}
.tsw #themeInput:checked + .slider .stars{transform:translateY(0);opacity:1}
.tsw #star-1{width:calc(var(--scale)*20px);top:calc(var(--scale)*2px);left:calc(var(--scale)*3px);animation-delay:.3s}
.tsw #star-2{width:calc(var(--scale)*6px);top:calc(var(--scale)*16px);left:calc(var(--scale)*3px)}
.tsw #star-3{width:calc(var(--scale)*12px);top:calc(var(--scale)*20px);left:calc(var(--scale)*10px);animation-delay:.6s}
.tsw #star-4{width:calc(var(--scale)*18px);top:0;left:calc(var(--scale)*18px);animation-delay:1.3s}
@keyframes tstar-twinkle{0%{transform:scale(1)}40%{transform:scale(1.2)}80%{transform:scale(.8)}100%{transform:scale(1)}}

/* ===== Sky (light mode only) ===== */
#pipSky{position:fixed;inset:0;z-index:0;overflow:hidden;background:linear-gradient(180deg,#2c8be6 0%,#5ab9ff 45%,#c4e8ff 100%)}
#petalCanvas{position:fixed;inset:0;z-index:2;pointer-events:none}
body:not(.light) #pipSky,body:not(.light) #petalCanvas{display:none}
.pip-sun{position:absolute;top:9vh;left:14vw;width:130px;height:130px;border-radius:50%;
  background:radial-gradient(circle at 34% 30%,#fffde7 0%,#fff176 22%,#ffc928 55%,#ff9800 100%);
  box-shadow:inset -14px -16px 26px rgba(205,95,0,.55),inset 10px 10px 18px rgba(255,255,230,.85),0 0 40px 14px rgba(255,235,120,.65),0 0 140px 50px rgba(255,214,70,.4),0 28px 40px rgba(255,140,0,.3);
  animation:sunPulse 3s ease-in-out infinite alternate}
.pip-sun:before{content:'';position:absolute;left:50%;top:50%;width:560px;height:560px;margin:-280px 0 0 -280px;border-radius:50%;z-index:-1;
  background:repeating-conic-gradient(rgba(255,255,255,.4) 0 3deg,transparent 3deg 15deg);
  -webkit-mask:radial-gradient(circle,transparent 24%,#000 26%,transparent 68%);mask:radial-gradient(circle,transparent 24%,#000 26%,transparent 68%);
  animation:raySpin 90s linear infinite}
@keyframes raySpin{to{transform:rotate(360deg)}}
@keyframes sunPulse{from{transform:scale(1)}to{transform:scale(1.05)}}
.pip-cw{position:absolute;left:0;animation:cloudDrift linear infinite}
@keyframes cloudDrift{from{transform:translateX(-360px)}to{transform:translateX(calc(100vw + 40px))}}
.pip-cloud{position:relative;width:200px;height:60px;border-radius:200px;transform-origin:left top;
  background:linear-gradient(#fff 10%,#eef4ff 55%,#c9d9f0 100%);
  box-shadow:inset 0 -12px 16px rgba(110,140,190,.5),inset 0 6px 8px #fff;filter:drop-shadow(0 16px 12px rgba(30,70,130,.32))}
.pip-cloud:before,.pip-cloud:after{content:'';position:absolute;border-radius:50%;width:100px;height:100px;top:-45px;left:22px;
  background:linear-gradient(#fff 15%,#eef4ff 60%,#d4e1f5 100%);box-shadow:inset 0 -10px 14px rgba(110,140,190,.4),inset 0 5px 7px #fff}
.pip-cloud:after{width:124px;height:124px;top:-62px;left:auto;right:20px}

/* ===== Light mode look ===== */
body.light{background:#5ab9ff;color:#12304f;font-family:'Patrick Hand','Caveat',cursive;font-size:19px}
body.light #stormy-canvas,body.light #stars-wrap,body.light .panel-stars{display:none}
body.light input,body.light select,body.light textarea,body.light button,
body.light #news-input,body.light #chat-input{font-family:'Patrick Hand','Caveat',cursive;font-size:19px}
body.light #code-input,body.light .bubble pre,body.light .bubble code,body.light #code-output pre{font-family:'Courier New',monospace;font-size:15px}
body.light h2,body.light .chat-welcome h3{font-family:'Caveat',cursive;font-size:2.2em;color:#b8325a;text-shadow:none}
body.light .chat-welcome h3{background:linear-gradient(90deg,#e0457b,#ff8a3d);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
body.light label{color:#1f4e8c;text-shadow:none;font-size:1.1em}
body.light .chat-welcome p{color:#2d4f78}
body.light .Button__text{font-family:'Caveat',cursive}
body.light .tab-inner{background:rgba(255,255,255,.3);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px)}
body.light .sidebar{background:rgba(255,255,255,.35);border-right:2px solid rgba(255,255,255,.8);box-shadow:0 0 30px rgba(40,100,180,.3)}
body.light .particle-btn{background:rgba(255,255,255,.55);border-color:rgba(255,255,255,.9);box-shadow:inset 0 2px 3px #fff,0 6px 12px rgba(30,80,150,.25)}
body.light .particle-btn span{color:#1f4e8c;text-transform:none;font-size:21px;font-weight:400}
body.light .particle-btn:hover{background:rgba(255,200,220,.65);box-shadow:0 0 20px rgba(255,140,170,.6)}
body.light .particle-btn:hover span,body.light .particle-btn.active span{color:#fff}
body.light .particle-btn.active{background:linear-gradient(135deg,#ff7aa2,#ffb347);border-color:#fff}
body.light input[type="text"],body.light textarea,body.light select{background:rgba(255,255,255,.62);border-color:rgba(60,120,200,.45);color:#12304f}
body.light input[type="text"]:focus,body.light textarea:focus,body.light select:focus{border-color:#ff7aa2;box-shadow:0 0 14px rgba(255,122,162,.5)}
body.light select option{background:#fff;color:#12304f}
body.light .chat-frame,body.light .output,body.light .composer{border-color:rgba(60,120,200,.4);background:rgba(255,255,255,.4)}
body.light .chat-frame::before,body.light .output::before{border-color:rgba(255,255,255,.9)}
body.light .composer:focus-within{border-color:#ff7aa2;box-shadow:0 0 14px rgba(255,122,162,.5)}
body.light .msg-row.user .bubble,body.light .avatar{background:linear-gradient(135deg,#ff7aa2,#ffb347)}
body.light .msg-row.assistant .avatar{background:rgba(255,255,255,.8);border-color:#ff7aa2;color:#d6336c}
body.light .bubble code{background:rgba(0,0,0,.08)}
body.light .bubble pre,body.light #code-output pre{color:#e8eefc}
body.light .typing i{background:#ff7aa2}
body.light .chat-container{scrollbar-color:rgba(255,122,162,.7) transparent}
body.light .cr-title{color:#b8325a}
body.light .cr-status.ok{color:#14683a;background:rgba(60,200,120,.25)}
body.light .cr-status.warn{color:#8a4b00;background:rgba(255,170,92,.3)}
body.light .cr-copy{background:rgba(255,255,255,.7);color:#b8325a;border-color:#ff7aa2}
body.light .generate-button{background-color:rgba(36,86,190,.62);color:#fff;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  box-shadow:inset 0 2px 3px rgba(255,255,255,.55),inset 0 -4px 8px rgba(0,0,40,.35),0 8px 18px rgba(30,70,160,.35)}
body.light .generate-button:hover{background-color:rgba(20,70,200,.8)}
body.light .generate-button span{font-family:'Patrick Hand',cursive;font-size:19px;font-weight:400}

/* ===== Voice buttons ===== */
.mic-btn{flex:none;width:46px;height:46px;border-radius:50%;border:2px solid rgba(221,132,72,.6);background:rgba(0,0,0,.5);color:#ffaa5c;font-size:20px;cursor:pointer;transition:all .2s}
.mic-btn:hover{background:rgba(221,132,72,.3);color:#fff}
.mic-btn.mini{width:38px;height:38px;font-size:16px}
.mic-btn.on{background:rgba(221,132,72,.55);color:#fff}
.mic-btn.rec{background:#e53935;border-color:#ff8a80;color:#fff;animation:micPulse 1s infinite}
.mic-btn:disabled{opacity:.5;cursor:not-allowed}
@keyframes micPulse{0%{box-shadow:0 0 0 0 rgba(229,57,53,.7)}100%{box-shadow:0 0 0 14px rgba(229,57,53,0)}}
body.light .mic-btn{background:rgba(255,255,255,.65);border-color:rgba(60,120,200,.5);color:#1f4e8c;box-shadow:inset 0 2px 3px #fff,0 5px 10px rgba(30,80,150,.25)}
body.light .mic-btn.on{background:linear-gradient(135deg,#ff7aa2,#ffb347);color:#fff}
body.light .mic-btn.rec{background:#e53935;color:#fff}
.voice-text{margin-top:6px;font-style:italic;opacity:.95}
.msg-row.user .bubble audio{display:block;max-width:100%;height:36px}
"""

PIP_BODY_HTML = r"""
<div id="pipSky"><div class="pip-sun"></div></div>
<canvas id="petalCanvas"></canvas>
<div class="theme-wrap">
  <label class="tsw" title="Colors: light / dark">
    <input id="themeInput" type="checkbox" checked>
    <div class="slider round"><div class="sun-moon" id="tswSunMoon"></div><div class="stars" id="tswStars"></div></div>
  </label>
  <div class="theme-label">Colors</div>
</div>
"""

PIP_JS = r"""
(function () {
  // ---------- build the toggle's little SVGs ----------
  const C = '<circle cx="50" cy="50" r="50"></circle>';
  const S = (id, cls) => '<svg id="' + id + '" class="' + cls + '" viewBox="0 0 100 100">' + C + '</svg>';
  let sm = '';
  for (let i = 1; i <= 3; i++) sm += S('moon-dot-' + i, 'moon-dot');
  for (let i = 1; i <= 3; i++) sm += S('light-ray-' + i, 'light-ray');
  for (let i = 1; i <= 3; i++) sm += S('cloud-' + i, 'cloud-dark');
  for (let i = 4; i <= 6; i++) sm += S('cloud-' + i, 'cloud-light');
  document.getElementById('tswSunMoon').innerHTML = sm;
  const star = '<path d="M 0 10 C 10 10,10 10 ,0 10 C 10 10 , 10 10 , 10 20 C 10 10 , 10 10 , 20 10 C 10 10 , 10 10 , 10 0 C 10 10,10 10 ,0 10 Z"></path>';
  let st = '';
  for (let i = 1; i <= 4; i++) st += '<svg id="star-' + i + '" class="star" viewBox="0 0 20 20">' + star + '</svg>';
  document.getElementById('tswStars').innerHTML = st;

  // ---------- 3D clouds ----------
  const sky = document.getElementById('pipSky');
  [[6, 1.1, 90, -10, .95, 0], [18, .8, 130, -60, .85, 0], [32, 1.5, 170, -30, .9, 0],
   [48, .6, 110, -80, .6, 1], [60, 1.2, 150, -100, .75, 1], [14, .5, 200, -150, .55, 1]]
  .forEach(c => {
    const w = document.createElement('div');
    w.className = 'pip-cw';
    w.style.cssText = 'top:' + c[0] + 'vh;animation-duration:' + c[2] + 's;animation-delay:' + c[3] + 's;opacity:' + c[4] + ';' + (c[5] ? 'filter:blur(1.5px);' : '');
    w.innerHTML = '<div class="pip-cloud" style="transform:scale(' + c[1] + ')"></div>';
    sky.appendChild(w);
  });

  // ---------- falling petals ----------
  const pc = document.getElementById('petalCanvas');
  const px = pc.getContext('2d');
  function sizeP() { pc.width = window.innerWidth; pc.height = window.innerHeight; }
  sizeP();
  window.addEventListener('resize', sizeP);
  let petalImg = new Image(), petalReady = false, petals = [], raf = 0, mouseX = 0;

  function fallbackPetal() {
    const c = document.createElement('canvas'); c.width = 60; c.height = 44;
    const g = c.getContext('2d');
    const gr = g.createRadialGradient(22, 18, 2, 30, 22, 30);
    gr.addColorStop(0, '#fff0f5'); gr.addColorStop(.6, '#ffb7c9'); gr.addColorStop(1, '#ff7fa3');
    g.fillStyle = gr; g.beginPath(); g.moveTo(4, 22);
    g.bezierCurveTo(14, -4, 48, 2, 58, 22); g.bezierCurveTo(48, 42, 14, 46, 4, 22); g.fill();
    petalImg = c; ready();
  }
  function ready() { petalReady = true; if (document.body.classList.contains('light')) startPetals(); }
  petalImg.onload = ready;
  petalImg.onerror = fallbackPetal;
  petalImg.src = 'https://djjjk9bjm164h.cloudfront.net/petal.png';

  function onMove(e) { mouseX = ((e.clientX !== undefined ? e.clientX : e.touches[0].clientX)) / window.innerWidth; }
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove);

  class Petal {
    constructor() {
      this.x = Math.random() * pc.width;
      this.y = Math.random() * pc.height * 2 - pc.height;
      this.w = 25 + Math.random() * 15; this.h = 20 + Math.random() * 10;
      this.opacity = this.w / 40; this.flip = Math.random();
      this.xSpeed = 1.5 + Math.random() * 2; this.ySpeed = 1 + Math.random();
      this.flipSpeed = Math.random() * 0.03;
    }
    draw() {
      if (this.y > pc.height || this.x > pc.width) {
        this.x = -(petalImg.width || 40);
        this.y = Math.random() * pc.height * 2 - pc.height;
        this.xSpeed = 1.5 + Math.random() * 2; this.ySpeed = 1 + Math.random();
        this.flip = Math.random();
      }
      px.globalAlpha = this.opacity;
      px.drawImage(petalImg, this.x, this.y,
        this.w * (0.6 + Math.abs(Math.cos(this.flip)) / 3),
        this.h * (0.8 + Math.abs(Math.sin(this.flip)) / 5));
    }
    animate() {
      this.x += this.xSpeed + mouseX * 5; this.y += this.ySpeed + mouseX * 2;
      this.flip += this.flipSpeed; this.draw();
    }
  }
  function loop() {
    // effects-off (lite) mode: skip drawing but keep the loop alive so petals
    // resume instantly when effects are turned back on
    if (document.body.classList.contains('lite')) {
      px.clearRect(0, 0, pc.width, pc.height);
    } else {
      px.clearRect(0, 0, pc.width, pc.height);
      petals.forEach(p => p.animate());
    }
    raf = requestAnimationFrame(loop);
  }
  function startPetals() {
    if (!petalReady || raf) return;
    if (!petals.length) for (let i = 0; i < 80; i++) petals.push(new Petal());
    loop();
  }
  function stopPetals() { cancelAnimationFrame(raf); raf = 0; px.clearRect(0, 0, pc.width, pc.height); }

  // ---------- theme switch ----------
  const input = document.getElementById('themeInput');
  function applyTheme(dark) {
    document.body.classList.toggle('light', !dark);
    if (dark) stopPetals(); else startPetals();
    try { localStorage.setItem('pipTheme', dark ? 'dark' : 'light'); } catch (e) {}
  }
  let saved = null;
  try { saved = localStorage.getItem('pipTheme'); } catch (e) {}
  input.checked = saved !== 'light';
  applyTheme(input.checked);
  input.addEventListener('change', () => applyTheme(input.checked));

  // ---------- voice messages ----------
  const composer = document.querySelector('#chat .composer');
  const head = document.querySelector('#chat .chat-head');
  const mic = document.createElement('button');
  mic.type = 'button'; mic.className = 'mic-btn'; mic.title = 'Record a voice message'; mic.textContent = '🎤';
  composer.insertBefore(mic, chatSendBtn);

  const speakBtn = document.createElement('button');
  speakBtn.type = 'button'; speakBtn.className = 'mic-btn mini'; speakBtn.title = 'Read replies out loud'; speakBtn.textContent = '🔊';
  const clearBtn = head.querySelector('button');
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;gap:10px;align-items:center';
  head.appendChild(wrap); wrap.appendChild(speakBtn); wrap.appendChild(clearBtn);

  let speakOn = false;
  speakBtn.addEventListener('click', () => {
    speakOn = !speakOn; speakBtn.classList.toggle('on', speakOn);
    if (!speakOn && window.speechSynthesis) speechSynthesis.cancel();
  });
  function speak(text) {
    if (!window.speechSynthesis) return;
    speechSynthesis.cancel();
    const clean = text.replace(/```[\s\S]*?```/g, ' code block. ').replace(/[*`#]/g, '');
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = navigator.language || 'en-US';
    speechSynthesis.speak(u);
  }

  let rec = null, chunks = [], stream = null, cancelled = false, tick = null;
  const defaultPh = chatInput.placeholder;

  function resetMicUI() {
    mic.classList.remove('rec'); mic.textContent = '🎤';
    chatInput.placeholder = defaultPh; clearInterval(tick);
  }
  async function startRec() {
    if (chatBusy) return;
    if (!navigator.mediaDevices || !window.MediaRecorder) { alert('Recording is not supported in this browser.'); return; }
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { alert('Please allow microphone access to send voice messages.'); return; }
    chunks = []; cancelled = false;
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'].find(t => MediaRecorder.isTypeSupported(t)) || '';
    rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
      if (!cancelled && blob.size > 800) sendVoice(blob);
    };
    rec.start();
    mic.classList.add('rec'); mic.textContent = '⏹';
    let s = 0;
    chatInput.placeholder = 'Recording… tap ⏹ to send (Esc cancels)';
    tick = setInterval(() => { s++; chatInput.placeholder = 'Recording ' + s + 's… tap ⏹ to send (Esc cancels)'; }, 1000);
  }
  function stopRec() { resetMicUI(); if (rec && rec.state === 'recording') rec.stop(); }
  mic.addEventListener('click', () => { (rec && rec.state === 'recording') ? stopRec() : startRec(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && rec && rec.state === 'recording') { cancelled = true; stopRec(); } });

  async function sendVoice(blob) {
    chatBusy = true; chatSendBtn.disabled = true; mic.disabled = true;
    const u = addRow('user');
    u.bubble.innerHTML = '<audio controls src="' + URL.createObjectURL(blob) + '"></audio><div class="voice-text">Transcribing…</div>';
    const bot = addRow('assistant');
    bot.bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    scrollChat();
    const done = () => { chatBusy = false; chatSendBtn.disabled = false; mic.disabled = false; };
    const label = u.bubble.querySelector('.voice-text');
    try {
      const fd = new FormData();
      const t = blob.type || '';
      fd.append('audio', blob, 'voice.' + (t.includes('mp4') ? 'm4a' : t.includes('ogg') ? 'ogg' : 'webm'));
      const res = await fetch('/api/voice', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.success) { label.textContent = ''; bot.bubble.textContent = 'Error: ' + data.error; done(); return; }
      label.textContent = '“' + data.transcript + '”';
      revealText(bot.bubble, data.response, () => { done(); if (speakOn) speak(data.response); });
    } catch (e) {
      label.textContent = ''; bot.bubble.textContent = 'Error: ' + e.message; done();
    }
  }
})();
"""


def _pip_get_html(self) -> str:
    html = _pip_orig_get_html(self)
    html = html.replace('</head>', PIP_HEAD + '<style>' + PIP_CSS + '</style></head>', 1)
    html = html.replace('<body>', '<body>' + PIP_BODY_HTML, 1)
    html = html.replace('</body>', '<script>' + PIP_JS + '</script></body>', 1)
    return html


def _pip_init(self):
    _pip_orig_init(self)
    self.voice = VoiceTranscriber()
    if FLASK_AVAILABLE:
        self._add_voice_route()


_pip_orig_get_html = WebUI.get_html
_pip_orig_init = WebUI.__init__
WebUI.get_html = _pip_get_html
WebUI.__init__ = _pip_init
WebUI._add_voice_route = _pip_add_voice_route

# ---------- light mode: pink buttons with yellow sparks + pastel blue leaf tab buttons ----------
PIP_CSS += r"""
/* action buttons (Send, Calculate, Check Code, Clear...) -> pink */
body.light .generate-button{
  background-color:rgba(255,105,160,.74);color:#fff;
  box-shadow:inset 0 2px 3px rgba(255,255,255,.75),inset 0 -5px 9px rgba(160,20,80,.4),0 8px 18px rgba(230,70,130,.4)}
body.light .generate-button:hover{
  background-color:rgba(255,80,145,.92);
  --generate-button-shadow-wide:rgba(255,90,160,.55);
  box-shadow:inset 0 2px 3px rgba(255,255,255,.85),inset 0 -5px 9px rgba(160,20,80,.45),0 10px 26px rgba(255,80,150,.6),0 0 0 3px rgba(255,190,215,.6)}
body.light .generate-button:before{background:#ff4f9a}
body.light .generate-button span:before{background-image:linear-gradient(to right,transparent,#ff69a8)}
body.light .generate-button svg.icon{fill:#ffd93d;
  filter:drop-shadow(0 1px 0 #e39b00) drop-shadow(0 2px 2px rgba(150,70,0,.55)) drop-shadow(0 0 5px rgba(255,230,90,.9))}
body.light .generate-button:after{
  background-image:
    radial-gradient(circle at 12% 30%,#fff6a8 0,rgba(255,246,168,0) 2.2px),
    radial-gradient(circle at 27% 72%,#ffd23f 0,rgba(255,210,63,0) 2px),
    radial-gradient(circle at 44% 22%,#fff6a8 0,rgba(255,246,168,0) 2.6px),
    radial-gradient(circle at 58% 76%,#ffd23f 0,rgba(255,210,63,0) 2px),
    radial-gradient(circle at 73% 28%,#fff6a8 0,rgba(255,246,168,0) 2.4px),
    radial-gradient(circle at 86% 68%,#ffd23f 0,rgba(255,210,63,0) 2.2px),
    radial-gradient(circle at 95% 25%,#fff6a8 0,rgba(255,246,168,0) 1.8px)}
body.light .generate-button svg.dots{fill:#ffd93d;filter:drop-shadow(0 0 3px rgba(255,220,70,.95))}
body.light .generate-button .stroke svg{stroke:#ffe98a}

/* sidebar tab buttons -> very light pastel blue, green leaves fall only on hover */
body.light .particle-btn{
  background:linear-gradient(135deg,#eaf6fd,#d3ecfa);border-color:#bfe1f5;
  box-shadow:inset 0 2px 3px #fff,inset 0 -4px 8px rgba(110,170,220,.22),0 6px 12px rgba(80,150,210,.25)}
body.light .particle-btn span{color:#2f6b9a}
body.light .particle-btn:hover{
  background:linear-gradient(135deg,#f4fbff,#c4e5f8);border-color:#ffb3cd;
  box-shadow:0 0 18px rgba(255,150,190,.5),inset 0 2px 3px #fff}
body.light .particle-btn.active{
  background:linear-gradient(135deg,#bfe4fa,#8ccbf0);border-color:#ff9bbd;
  box-shadow:0 0 18px rgba(255,140,185,.55),inset 0 2px 3px rgba(255,255,255,.9)}
body.light .particle-btn:hover span,body.light .particle-btn.active span{color:#17496f}
body.light .particle-btn canvas:not(.pip-btn-leaves){display:none}
body:not(.light) .pip-btn-leaves{display:none}
.pip-btn-leaves{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
"""

PIP_JS += r"""
(function () {
  const COLORS = [['#9be3a2', '#4fbf72'], ['#7fd68a', '#3fa860'], ['#b4ecb0', '#5cc47a'], ['#6fcf7f', '#379653']];
  const btns = Array.from(document.querySelectorAll('.particle-btn'));

  function newLeaf(w, h, startAbove) {
    return { x: Math.random() * w, y: startAbove ? -Math.random() * h * 1.2 - 8 : -10 - Math.random() * 20,
             s: 6 + Math.random() * 5, vy: 0.7 + Math.random() * 0.9, sway: Math.random() * 6.28,
             swaySpeed: 0.04 + Math.random() * 0.04, rot: Math.random() * 6.28, vr: (Math.random() - 0.5) * 0.06,
             c: COLORS[Math.floor(Math.random() * COLORS.length)], alive: true };
  }
  function drawLeaf(g, l) {
    const s = l.s;
    g.save(); g.translate(l.x, l.y); g.rotate(l.rot);
    const gr = g.createLinearGradient(-s, 0, s, 0);
    gr.addColorStop(0, l.c[0]); gr.addColorStop(1, l.c[1]);
    g.fillStyle = gr; g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 0.7;
    g.beginPath(); g.moveTo(-s, 0);
    g.bezierCurveTo(-s * .4, -s * .95, s * .6, -s * .85, s * 1.1, 0);
    g.bezierCurveTo(s * .6, s * .85, -s * .4, s * .95, -s, 0);
    g.fill(); g.stroke();
    g.strokeStyle = 'rgba(25,100,55,.55)'; g.lineWidth = 0.9;
    g.beginPath(); g.moveTo(-s, 0); g.lineTo(s * .95, 0); g.stroke();
    g.restore();
  }

  const items = btns.map(b => {
    const c = document.createElement('canvas');
    c.className = 'pip-btn-leaves';
    b.insertBefore(c, b.firstChild);
    const it = { b, c, g: c.getContext('2d'), leaves: [], hover: false };
    const start = () => {
      it.hover = true;
      const w = b.offsetWidth, h = b.offsetHeight;
      it.leaves = [];
      for (let i = 0; i < 12; i++) it.leaves.push(newLeaf(w, h, true));
    };
    const stop = () => { it.hover = false; };   // leaves already falling finish their fall
    b.addEventListener('pointerenter', start);
    b.addEventListener('pointerleave', stop);
    return it;
  });

  function frame() {
    if (document.body.classList.contains('light')) {
      items.forEach(it => {
        if (!it.leaves.length) return;
        const w = it.b.offsetWidth, h = it.b.offsetHeight;
        if (!w || !h) return;
        if (it.c.width !== w || it.c.height !== h) { it.c.width = w; it.c.height = h; }
        it.g.clearRect(0, 0, w, h);
        it.leaves.forEach(l => {
          l.y += l.vy; l.sway += l.swaySpeed; l.x += Math.sin(l.sway) * 0.35; l.rot += l.vr;
          if (l.y > h + 12) {
            if (it.hover) Object.assign(l, newLeaf(w, h, false));
            else l.alive = false;
          }
          if (l.alive) drawLeaf(it.g, l);
        });
        it.leaves = it.leaves.filter(l => l.alive);
        if (!it.leaves.length) it.g.clearRect(0, 0, w, h);
      });
    }
    requestAnimationFrame(frame);
  }
  frame();
})();
"""
# ============================================================================
# IMAGE ANALYSIS: objects + brain MRI tumor detection
# ============================================================================
import io
import base64
import hashlib
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv not installed. Run: pip install opencv-python-headless")


class ImageAnalyzer:
    """Object detection + captioning for normal photos, and
    classification + localization + size measurement for brain MRI."""

    OBJECT_MODEL = os.getenv("OBJECT_MODEL", "facebook/detr-resnet-101")
    CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
    CLIP_MODEL = "openai/clip-vit-large-patch14"

    # Tried in order; every one that loads and has >= 3 known classes joins the
    # ensemble (max 3). Override with env MRI_MODELS="repo1,repo2".
    MRI_CANDIDATES = [
        "sartajbhuvaji/brain-tumor-classification",
        "DunnBC22/vit-base-patch16-224-in21k_brain_tumor_diagnosis",
        "Hemg/Brain-Tumor-Classification",
    ]

    TYPE_LABELS = {
        "an MRI scan of a human brain": "brain_mri",
        "an MRI scan of a knee, spine or other body part": "other_mri",
        "a CT scan": "ct",
        "an X-ray image": "xray",
        "an ultrasound image": "ultrasound",
        "a normal color photograph": "photo",
        "a drawing, screenshot or illustration": "drawing",
    }
    PLANE_LABELS = {
        "an axial MRI slice of the brain, seen from above": "axial",
        "a sagittal MRI slice of the brain, seen from the side": "sagittal",
        "a coronal MRI slice of the brain, seen from the front": "coronal",
    }
    TUMOR_NAMES = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary tumor (adenoma)",
    }
    MAX_SIDE = 2000

    def __init__(self):
        self._pipes = {}
        self._mri_models = None
        self._lock = threading.Lock()
        self.device = -1
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0
        except Exception:
            pass

    # ------------------------------------------------------------ loading
    def _pipe(self, key, task, model, **kw):
        if key not in self._pipes:
            print(f"Loading {model} ...")
            self._pipes[key] = pipeline(task, model=model, device=self.device, **kw)
        return self._pipes[key]

    def _load_mri_models(self):
        if self._mri_models is not None:
            return self._mri_models
        names = os.getenv("MRI_MODELS")
        names = [n.strip() for n in names.split(",")] if names else self.MRI_CANDIDATES
        loaded = []
        for name in names:
            if len(loaded) >= 3:
                break
            try:
                clf = pipeline("image-classification", model=name, device=self.device)
                labels = [self._norm_label(l) for l in clf.model.config.id2label.values()]
                if len({l for l in labels if l}) < 3:
                    print(f"Skipping {name}: labels not recognised ({list(clf.model.config.id2label.values())})")
                    continue
                loaded.append((name, clf))
                print(f"✓ Loaded tumor model {name}")
            except Exception as e:
                print(f"Could not load {name}: {str(e)[:120]}")
        self._mri_models = loaded
        return loaded

    @staticmethod
    def _norm_label(label):
        l = str(label).lower().replace("_", " ").replace("-", " ")
        if any(k in l for k in ("no tumor", "notumor", "no tumour", "normal", "healthy", "negative")):
            return "no tumor"
        for k in ("glioma", "meningioma", "pituitary"):
            if k in l:
                return k
        return None

    def _class_probs(self, out):
        probs = {}
        for d in out:
            k = self._norm_label(d["label"])
            if k:
                probs[k] = probs.get(k, 0.0) + float(d["score"])
        s = sum(probs.values())
        return {k: v / s for k, v in probs.items()} if s > 0 else {}

    # ------------------------------------------------------------ input
    def _load_image(self, data, filename, spacing):
        note = None
        if filename and filename.lower().endswith((".dcm", ".dicom")):
            try:
                import pydicom
                ds = pydicom.dcmread(io.BytesIO(data))
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
                    arr = arr[arr.shape[0] // 2]
                lo, hi = np.percentile(arr, (0.5, 99.5))
                arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, -1)
                img = Image.fromarray(arr.astype(np.uint8)[..., :3])
                if not spacing and hasattr(ds, "PixelSpacing"):
                    spacing = float(ds.PixelSpacing[0])
                    note = f"Pixel spacing read from the DICOM file: {spacing:.3f} mm/pixel."
            except Exception as e:
                raise ValueError(f"Could not read the DICOM file: {e}")
        else:
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGB")
        scale = 1.0
        if max(img.size) > self.MAX_SIDE:
            scale = max(img.size) / self.MAX_SIDE
            img = img.resize((int(img.width / scale), int(img.height / scale)), Image.LANCZOS)
        eff_spacing = spacing * scale if spacing else None
        return img, eff_spacing, note

    # ------------------------------------------------------------ main
    def analyze(self, data, filename="", mode="auto", spacing=None):
        if not (TRANSFORMERS_AVAILABLE and CV2_AVAILABLE):
            return {"success": False, "error": "transformers and opencv-python-headless are required."}
        with self._lock:
            try:
                img, spacing, dicom_note = self._load_image(data, filename, spacing)
            except Exception as e:
                return {"success": False, "error": str(e)}

            result = {"success": True, "width": img.width, "height": img.height, "notes": []}
            if dicom_note:
                result["notes"].append(dicom_note)

            kind, itype = mode, None
            if mode == "auto" or mode == "mri":
                itype = self._image_type(img)
                result["image_type"] = itype
                if mode == "auto":
                    kind = "mri" if itype["type"] == "brain_mri" else "objects"
                    if itype["type"] in ("other_mri", "ct", "xray", "ultrasound"):
                        result["notes"].append(
                            f"This looks like {itype['label']}. The tumor model only supports brain MRI, "
                            "so I analysed it as a general image. Switch the mode to 'Brain MRI' to force tumor analysis.")
            if kind == "mri":
                result.update(self._analyze_mri(img, spacing, result))
            else:
                result.update(self._analyze_objects(img))
            return result

    # ------------------------------------------------------------ CLIP
    def _image_type(self, img):
        clip = self._pipe("clip", "zero-shot-image-classification", self.CLIP_MODEL)
        out = clip(img, candidate_labels=list(self.TYPE_LABELS))
        top = out[0]
        res = {"type": self.TYPE_LABELS[top["label"]], "label": top["label"], "confidence": round(top["score"] * 100, 1)}
        if res["type"] == "brain_mri":
            p = clip(img, candidate_labels=list(self.PLANE_LABELS))[0]
            res["plane"] = self.PLANE_LABELS[p["label"]]
            res["plane_confidence"] = round(p["score"] * 100, 1)
        return res

    # ------------------------------------------------------------ objects
    @staticmethod
    def _color(label):
        h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
        return (60 + h % 190, 60 + (h >> 8) % 190, 60 + (h >> 16) % 190)

    @staticmethod
    def _font(size):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _position(cx, cy, W, H):
        v = "top" if cy < H * 0.34 else "bottom" if cy > H * 0.66 else "middle"
        h = "left" if cx < W * 0.34 else "right" if cx > W * 0.66 else "center"
        return f"{v}-{h}" if (v, h) != ("middle", "center") else "center"

    @staticmethod
    def _size_class(pct):
        return "small" if pct < 2 else "medium" if pct < 15 else "large"

    def _to_data_url(self, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _analyze_objects(self, img):
        W, H = img.size
        det = self._pipe("det", "object-detection", self.OBJECT_MODEL)
        dets = [d for d in det(img) if d["score"] >= 0.35]
        dets.sort(key=lambda d: -d["score"])

        caption = None
        try:
            cap = self._pipe("cap", "image-to-text", self.CAPTION_MODEL)
            caption = cap(img, max_new_tokens=60)[0]["generated_text"].strip()
        except Exception as e:
            print("Caption error:", e)

        draw_img = img.copy()
        d = ImageDraw.Draw(draw_img)
        font = self._font(max(13, int(min(W, H) / 38)))
        lw = max(2, int(min(W, H) / 250))
        objects, counts = [], {}
        for o in dets:
            b = o["box"]
            x0, y0, x1, y1 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
            w, h = x1 - x0, y1 - y0
            pct = w * h / (W * H) * 100
            label = o["label"]
            counts[label] = counts.get(label, 0) + 1
            objects.append({
                "label": label, "confidence": round(o["score"] * 100, 1),
                "width_px": int(w), "height_px": int(h), "area_percent": round(pct, 2),
                "size_class": self._size_class(pct),
                "position": self._position((x0 + x1) / 2, (y0 + y1) / 2, W, H),
            })
            col = self._color(label)
            d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
            text = f"{label} {o['score'] * 100:.0f}% | {int(w)}x{int(h)}px"
            tb = d.textbbox((0, 0), text, font=font)
            th = tb[3] - tb[1] + 6
            ty = y0 - th if y0 - th > 0 else y0
            d.rectangle([x0, ty, x0 + tb[2] - tb[0] + 8, ty + th], fill=col)
            d.text((x0 + 4, ty + 2), text, fill=(255, 255, 255), font=font)

        return {"kind": "objects", "caption": caption, "objects": objects,
                "object_counts": counts, "annotated_image": self._to_data_url(draw_img)}

    # ------------------------------------------------------------ MRI
    @staticmethod
    def _largest_component(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if n <= 1:
            return np.zeros_like(mask, dtype=bool)
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == i

    def _brain_mask(self, img):
        from scipy.ndimage import binary_fill_holes
        gray = cv2.GaussianBlur(np.array(img.convert("L")), (5, 5), 0)
        m = gray > max(15, int(gray.max() * 0.08))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
        m = self._largest_component(binary_fill_holes(m))
        return m

    def _ensemble(self, img, models):
        probs_all = []
        for name, clf in models:
            out = clf(img, top_k=None)
            p = self._class_probs(out)
            if p:
                probs_all.append(p)
        keys = {k for p in probs_all for k in p}
        avg = {k: sum(p.get(k, 0) for p in probs_all) / len(probs_all) for k in keys}
        s = sum(avg.values())
        return {k: v / s for k, v in avg.items()}

    def _heatmap(self, img, clf, target):
        """Occlusion sensitivity: hide a patch, see how much the tumor score drops."""
        S, win, stride = 224, 56, 16
        base = img.resize((S, S), Image.BICUBIC)
        arr = np.array(base)
        fill = arr.mean(axis=(0, 1)).astype(np.uint8)
        base_p = self._class_probs(clf(base, top_k=None)).get(target, 0.0)
        coords = [(y, x) for y in range(0, S - win + 1, stride) for x in range(0, S - win + 1, stride)]
        batch = []
        for y, x in coords:
            a = arr.copy()
            a[y:y + win, x:x + win] = fill
            batch.append(Image.fromarray(a))
        outs = clf(batch, top_k=None, batch_size=32)
        heat = np.zeros((S, S), np.float32)
        cnt = np.zeros((S, S), np.float32)
        for (y, x), o in zip(coords, outs):
            drop = max(0.0, base_p - self._class_probs(o).get(target, 0.0))
            heat[y:y + win, x:x + win] += drop
            cnt[y:y + win, x:x + win] += 1
        heat /= np.maximum(cnt, 1)
        return cv2.GaussianBlur(heat, (0, 0), 6)

    def _analyze_mri(self, img, spacing, result):
        W, H = img.size
        models = self._load_mri_models()
        if not models:
            return {"kind": "mri", "error_detail": "No tumor model could be loaded.",
                    "tumor": None, "annotated_image": self._to_data_url(img),
                    "notes": result["notes"] + [
                        "No brain-tumor model could be downloaded. Set the env variable MRI_MODELS to a "
                        "Hugging Face image-classification repo with glioma / meningioma / pituitary / notumor labels."]}

        probs = self._ensemble(img, models)
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        top, conf = ranked[0]
        notes = list(result["notes"])
        tumor = {
            "found": top != "no tumor",
            "type": "No tumor detected" if top == "no tumor" else self.TUMOR_NAMES.get(top, top.title()),
            "confidence": round(conf * 100, 1),
            "probabilities": {("No tumor" if k == "no tumor" else self.TUMOR_NAMES.get(k, k)): round(v * 100, 1) for k, v in ranked},
            "models_used": [n for n, _ in models],
            "size": None, "location": None,
        }
        if conf < 0.6:
            notes.append("Confidence is low. The models disagree or the image is unusual, so treat this result with extra caution.")
        if len(models) == 1:
            notes.append("Only one tumor model loaded, so there is no ensemble cross-check.")

        annotated = img.copy()
        if tumor["found"]:
            heat = self._heatmap(img, models[0][1], top)
            if heat.max() < 0.02:
                notes.append("The tumor was classified, but its exact location could not be located reliably, so no outline or size is given.")
            else:
                brain = self._brain_mask(img)
                hu = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
                region = (hu >= 0.45 * hu.max()) & brain
                region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
                region = self._largest_component(region)
                area = int(region.sum())
                if area > 20:
                    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = max(cnts, key=cv2.contourArea)
                    (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(cnt)
                    length, width = max(rw, rh), min(rw, rh)
                    eq = 2 * (area / np.pi) ** 0.5
                    ys, xs = np.where(region)
                    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                    brain_area = max(int(brain.sum()), 1)
                    size = {
                        "length_px": round(length, 1), "width_px": round(width, 1),
                        "equivalent_diameter_px": round(eq, 1), "area_px": area,
                        "percent_of_brain_slice": round(area / brain_area * 100, 2),
                        "bbox": [x0, y0, x1, y1],
                    }
                    if spacing:
                        size.update({
                            "length_mm": round(length * spacing, 1), "width_mm": round(width * spacing, 1),
                            "equivalent_diameter_mm": round(eq * spacing, 1),
                            "area_mm2": round(area * spacing * spacing, 1),
                        })
                    else:
                        notes.append("No pixel spacing was given, so sizes are in pixels. Enter mm/pixel (or upload a DICOM file) to get millimetres.")
                    tumor["size"] = size

                    bys, bxs = np.where(brain)
                    bx0, bx1, by0, by1 = bxs.min(), bxs.max(), bys.min(), bys.max()
                    cx, cy = xs.mean(), ys.mean()
                    fx = (cx - bx0) / max(bx1 - bx0, 1)
                    fy = (cy - by0) / max(by1 - by0, 1)
                    hor = "left" if fx < 0.4 else "right" if fx > 0.6 else "central"
                    ver = "upper" if fy < 0.4 else "lower" if fy > 0.6 else "middle"
                    tumor["location"] = (f"{ver}-{hor} part of the brain slice as displayed (image left/right; "
                                         "patient side depends on the scan's orientation convention)")

                    # draw highlight
                    base = annotated.convert("RGBA")
                    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
                    od = ImageDraw.Draw(ov)
                    pts = [tuple(p[0]) for p in cnt]
                    od.polygon(pts, fill=(255, 40, 40, 90))
                    base = Image.alpha_composite(base, ov).convert("RGB")
                    d = ImageDraw.Draw(base)
                    lw = max(2, int(min(W, H) / 200))
                    d.line(pts + [pts[0]], fill=(255, 40, 40), width=lw)
                    d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(255, 235, 0), width=max(1, lw - 1))
                    label = f"{tumor['type']} {tumor['confidence']:.0f}%"
                    if spacing:
                        label += f" | {size['length_mm']}x{size['width_mm']} mm"
                    else:
                        label += f" | {int(length)}x{int(width)} px"
                    font = self._font(max(13, int(min(W, H) / 30)))
                    tb = d.textbbox((0, 0), label, font=font)
                    th = tb[3] - tb[1] + 8
                    ty = y0 - th - 6 if y0 - th - 6 > 0 else y1 + 6
                    d.rectangle([x0, ty, x0 + tb[2] - tb[0] + 10, ty + th], fill=(200, 20, 20))
                    d.text((x0 + 5, ty + 3), label, fill=(255, 255, 255), font=font)
                    annotated = base
                else:
                    notes.append("The located region was too small to measure.")
        notes.append("Tumor extent is estimated from model attention (occlusion map), not a pixel-perfect segmentation. "
                     "Tumor grade, stage or exact boundaries cannot be determined from one 2D image.")
        return {"kind": "mri", "tumor": tumor, "notes": notes,
                "annotated_image": self._to_data_url(annotated),
                "disclaimer": "This is an AI screening aid, not a medical diagnosis. A radiologist or doctor must review the scan."}


def _img_add_route(self):
    @self.app.route('/api/analyze_image', methods=['POST'])
    def analyze_image():
        f = request.files.get('image')
        if not f:
            return jsonify({'success': False, 'error': 'No image received.'}), 400
        mode = request.form.get('mode', 'auto')
        try:
            spacing = float(request.form.get('spacing') or 0) or None
        except ValueError:
            spacing = None
        try:
            res = self.image_analyzer.analyze(f.read(), f.filename or "", mode, spacing)
            return jsonify(res), (200 if res.get('success') else 400)
        except Exception as e:
            print(f"Image error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


_img_prev_init = WebUI.__init__
_img_prev_get_html = WebUI.get_html


def _img_init(self):
    _img_prev_init(self)
    self.image_analyzer = ImageAnalyzer()
    if FLASK_AVAILABLE:
        _img_add_route(self)


IMG_TAB_BUTTON = '<button class="particle-btn" data-tab="image"><span>🖼️ Image Analysis</span><canvas></canvas></button>'

IMG_PANEL = r"""
<!-- IMAGE ANALYSIS -->
<div id="image" class="tab-content">
  <div class="tab-inner">
    <h2>Image Analysis</h2>
    <div class="field-row" style="grid-template-columns:1fr 1fr 1fr">
      <div class="input-group">
        <label>Mode</label>
        <select id="img-mode">
          <option value="auto">Auto-detect</option>
          <option value="mri">Brain MRI (tumor analysis)</option>
          <option value="objects">Objects / general photo</option>
        </select>
      </div>
      <div class="input-group">
        <label>Pixel spacing (mm/pixel, optional)</label>
        <input type="text" id="img-spacing" placeholder="e.g. 0.94">
      </div>
      <div class="input-group">
        <label>Image (JPG, PNG, DICOM)</label>
        <input type="file" id="img-file" accept="image/*,.dcm,.dicom" style="width:100%;padding:10px;color:inherit">
      </div>
    </div>
    <img id="img-preview" alt="" style="display:none;max-width:100%;max-height:300px;border-radius:14px;margin-bottom:16px">
    <button class="generate-button" onclick="analyzeImage()"><span>Analyze Image</span></button>
    <div class="output" id="img-output"></div>
  </div>
</div>
"""

IMG_CSS = r"""
#img-output{white-space:normal}
#img-output img.result{max-width:100%;border-radius:14px;margin:10px 0;border:2px solid rgba(255,255,255,.3)}
#img-output table{width:100%;border-collapse:collapse;margin:8px 0}
#img-output th,#img-output td{padding:6px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.2)}
#img-output ul{margin:0;padding-left:22px}
#img-output li{margin-bottom:6px}
.img-bar{height:10px;border-radius:6px;background:rgba(255,255,255,.15);overflow:hidden;min-width:120px}
.img-bar i{display:block;height:100%;background:linear-gradient(90deg,#a855f7,#e879f9)}
.img-disc{margin-top:14px;padding:10px 14px;border-radius:12px;background:rgba(255,80,80,.15);border:1px solid rgba(255,80,80,.5)}
"""

IMG_JS = r"""
(function () {
  const fileIn = document.getElementById('img-file');
  const prev = document.getElementById('img-preview');
  fileIn.addEventListener('change', () => {
    const f = fileIn.files[0];
    if (!f) return;
    if (/\.(dcm|dicom)$/i.test(f.name)) { prev.style.display = 'none'; return; }
    prev.src = URL.createObjectURL(f);
    prev.style.display = 'block';
  });

  const list = a => '<ul>' + a.map(x => '<li>' + escapeHtml(String(x)) + '</li>').join('') + '</ul>';
  const title = t => '<div class="cr-title">' + t + '</div>';

  function renderMri(d) {
    let h = '';
    const t = d.tumor;
    if (d.image_type) {
      h += title('🧠 Image type') + '<p>' + escapeHtml(d.image_type.label) + ' (' + d.image_type.confidence + '%)' +
           (d.image_type.plane ? ', plane: <b>' + d.image_type.plane + '</b>' : '') + '</p>';
    }
    if (!t) { h += list(d.notes || []); return h; }
    h += '<div class="cr-status ' + (t.found ? 'warn' : 'ok') + '">' +
         (t.found ? '⚠️ Tumor detected: ' : '✅ ') + escapeHtml(t.type) + ' (' + t.confidence + '% confidence)</div>';
    h += '<img class="result" src="' + d.annotated_image + '">';
    h += title('📊 Class probabilities') + '<table>' +
         Object.entries(t.probabilities).map(([k, v]) =>
           '<tr><td>' + escapeHtml(k) + '</td><td>' + v + '%</td><td><div class="img-bar"><i style="width:' + v + '%"></i></div></td></tr>').join('') +
         '</table>';
    if (t.size) {
      const s = t.size, mm = s.length_mm !== undefined;
      h += title('📏 Tumor size') + '<table>' +
        '<tr><td>Length</td><td>' + (mm ? s.length_mm + ' mm' : s.length_px + ' px') + '</td></tr>' +
        '<tr><td>Width</td><td>' + (mm ? s.width_mm + ' mm' : s.width_px + ' px') + '</td></tr>' +
        '<tr><td>Equivalent diameter</td><td>' + (mm ? s.equivalent_diameter_mm + ' mm' : s.equivalent_diameter_px + ' px') + '</td></tr>' +
        '<tr><td>Area</td><td>' + (mm ? s.area_mm2 + ' mm²' : s.area_px + ' px²') + '</td></tr>' +
        '<tr><td>Share of brain slice</td><td>' + s.percent_of_brain_slice + '%</td></tr></table>';
    }
    if (t.location) h += title('📍 Location') + '<p>' + escapeHtml(t.location) + '</p>';
    h += title('🤖 Models used') + list(t.models_used);
    if (d.notes && d.notes.length) h += title('📝 Notes') + list(d.notes);
    if (d.disclaimer) h += '<div class="img-disc">⚕️ ' + escapeHtml(d.disclaimer) + '</div>';
    return h;
  }

  function renderObjects(d) {
    let h = '';
    if (d.image_type) h += title('🔎 Image type') + '<p>' + escapeHtml(d.image_type.label) + ' (' + d.image_type.confidence + '%)</p>';
    if (d.caption) h += title('👁️ What I see') + '<p>' + escapeHtml(d.caption) + '</p>';
    h += '<img class="result" src="' + d.annotated_image + '">';
    const c = Object.entries(d.object_counts || {});
    if (c.length) h += title('📦 Objects found') + '<p>' + c.map(([k, v]) => v + ' × ' + escapeHtml(k)).join(', ') + '</p>';
    if (d.objects && d.objects.length) {
      h += '<table><tr><th>Object</th><th>Conf.</th><th>Size (px)</th><th>% of image</th><th>Class</th><th>Position</th></tr>' +
        d.objects.map(o => '<tr><td>' + escapeHtml(o.label) + '</td><td>' + o.confidence + '%</td><td>' + o.width_px + '×' + o.height_px +
          '</td><td>' + o.area_percent + '%</td><td>' + o.size_class + '</td><td>' + o.position + '</td></tr>').join('') + '</table>';
    } else {
      h += '<p>No objects detected with high confidence.</p>';
    }
    if (d.notes && d.notes.length) h += title('📝 Notes') + list(d.notes);
    return h;
  }

  window.analyzeImage = async function () {
    const out = document.getElementById('img-output');
    const f = fileIn.files[0];
    if (!f) { out.textContent = 'Please choose an image first.'; return; }
    out.textContent = 'Analyzing... (the first run downloads the models and can take a few minutes)';
    const fd = new FormData();
    fd.append('image', f);
    fd.append('mode', document.getElementById('img-mode').value);
    fd.append('spacing', document.getElementById('img-spacing').value);
    try {
      const res = await fetch('/api/analyze_image', { method: 'POST', body: fd });
      const d = await res.json();
      if (!d.success) { out.textContent = d.error || 'Something went wrong.'; return; }
      out.innerHTML = d.kind === 'mri' ? renderMri(d) : renderObjects(d);
      out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
      out.textContent = 'Error: ' + e.message;
    }
  };
})();
"""


def _img_get_html(self) -> str:
    html = _img_prev_get_html(self)
    html = html.replace('<button class="particle-btn" data-tab="news">',
                        IMG_TAB_BUTTON + '<button class="particle-btn" data-tab="news">', 1)
    html = html.replace('<!-- FAKE NEWS -->', IMG_PANEL + '<!-- FAKE NEWS -->', 1)
    html = html.replace('</head>', '<style>' + IMG_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + IMG_JS + '</script></body>', 1)
    return html


WebUI.__init__ = _img_init
WebUI.get_html = _img_get_html

# ============================================================================
# SPEED + LIVE PROGRESS PATCH FOR IMAGE ANALYSIS
# ============================================================================
import uuid
import time

# ---- 1) lighter, faster models (set env vars to go back to the big ones) ----
ImageAnalyzer.CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
ImageAnalyzer.CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
ImageAnalyzer.OBJECT_MODEL = "facebook/detr-resnet-101"
ImageAnalyzer.MAX_SIDE = 1280


# ---- 2) fp16 on GPU ----
def _fast_pipe(self, key, task, model, **kw):
    if key not in self._pipes:
        import torch
        if self.device == 0:
            kw.setdefault("torch_dtype", torch.float16)
        print(f"Loading {model} ...")
        self._pipes[key] = pipeline(task, model=model, device=self.device, **kw)
    return self._pipes[key]


def _fast_load_mri_models(self):
    if self._mri_models is not None:
        return self._mri_models
    import torch
    names = os.getenv("MRI_MODELS")
    names = [n.strip() for n in names.split(",")] if names else self.MRI_CANDIDATES
    loaded = []
    for name in names:
        if len(loaded) >= 3:
            break
        try:
            kw = {"torch_dtype": torch.float16} if self.device == 0 else {}
            clf = pipeline("image-classification", model=name, device=self.device, **kw)
            labels = [self._norm_label(l) for l in clf.model.config.id2label.values()]
            if len({l for l in labels if l}) < 3:
                print(f"Skipping {name}: labels not recognised")
                continue
            loaded.append((name, clf))
            print(f"✓ Loaded tumor model {name}")
        except Exception as e:
            print(f"Could not load {name}: {str(e)[:120]}")
    self._mri_models = loaded
    return loaded


# ---- 3) occlusion map: 36 passes in ONE batch (was ~121) ----
def _fast_heatmap(self, img, clf, target):
    S, win, stride = 224, 64, 32
    base = img.resize((S, S), Image.BICUBIC)
    arr = np.array(base)
    fill = arr.mean(axis=(0, 1)).astype(np.uint8)
    coords = [(y, x) for y in range(0, S - win + 1, stride) for x in range(0, S - win + 1, stride)]
    batch = [base]
    for y, x in coords:
        a = arr.copy()
        a[y:y + win, x:x + win] = fill
        batch.append(Image.fromarray(a))
    outs = clf(batch, top_k=None, batch_size=64)
    base_p = self._class_probs(outs[0]).get(target, 0.0)
    heat = np.zeros((S, S), np.float32)
    cnt = np.zeros((S, S), np.float32)
    for (y, x), o in zip(coords, outs[1:]):
        drop = max(0.0, base_p - self._class_probs(o).get(target, 0.0))
        heat[y:y + win, x:x + win] += drop
        cnt[y:y + win, x:x + win] += 1
    heat /= np.maximum(cnt, 1)
    return cv2.GaussianBlur(heat, (0, 0), 10)


ImageAnalyzer._pipe = _fast_pipe
ImageAnalyzer._load_mri_models = _fast_load_mri_models
ImageAnalyzer._heatmap = _fast_heatmap


# ---- 4) progress reporting (stage name + percent) ----
def _stage(name, pct):
    def deco(fn):
        def wrapper(self, *a, **k):
            cb = getattr(self, "_cb", None)
            if cb:
                cb(name, pct)
            return fn(self, *a, **k)
        return wrapper
    return deco


ImageAnalyzer._load_image = _stage("Reading the image", 6)(ImageAnalyzer._load_image)
ImageAnalyzer._image_type = _stage("Identifying the image type", 15)(ImageAnalyzer._image_type)
ImageAnalyzer._analyze_objects = _stage("Detecting objects and describing the scene", 40)(ImageAnalyzer._analyze_objects)
ImageAnalyzer._load_mri_models = _stage("Preparing the tumor models", 30)(ImageAnalyzer._load_mri_models)
ImageAnalyzer._ensemble = _stage("Classifying the tumor type", 50)(ImageAnalyzer._ensemble)
ImageAnalyzer._heatmap = _stage("Locating the tumor and measuring it", 72)(ImageAnalyzer._heatmap)
ImageAnalyzer._brain_mask = _stage("Outlining and drawing results", 88)(ImageAnalyzer._brain_mask)


# ---- 5) cache + preload ----
_orig_analyze = ImageAnalyzer.analyze


def _analyze_cached(self, data, filename="", mode="auto", spacing=None, progress=None):
    key = hashlib.md5(data).hexdigest() + f"|{mode}|{spacing}"
    if key in self._cache:
        return self._cache[key]
    with self._job_lock:                 # one analysis at a time; others show "Queued"
        self._cb = progress
        try:
            res = _orig_analyze(self, data, filename, mode, spacing)
        finally:
            self._cb = None
    if res.get("success"):
        if len(self._cache) > 20:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = res
    return res


def _preload(self):
    with self._lock:
        try:
            self._pipe("clip", "zero-shot-image-classification", self.CLIP_MODEL)
            self._pipe("det", "object-detection", self.OBJECT_MODEL)
            self._pipe("cap", "image-to-text", self.CAPTION_MODEL)
            self._load_mri_models()
            print("✓ Image models preloaded, first analysis will be fast")
        except Exception as e:
            print("Image model preload problem:", str(e)[:150])


ImageAnalyzer.analyze = _analyze_cached
ImageAnalyzer._preload = _preload


# ---- 6) background jobs + status endpoint ----
IMAGE_JOBS = {}


def _fast_add_routes(self):
    @self.app.route('/api/analyze_start', methods=['POST'])
    def analyze_start():
        f = request.files.get('image')
        if not f:
            return jsonify({'success': False, 'error': 'No image received.'}), 400
        mode = request.form.get('mode', 'auto')
        try:
            spacing = float(request.form.get('spacing') or 0) or None
        except ValueError:
            spacing = None
        data, fname = f.read(), f.filename or ""
        jid = uuid.uuid4().hex
        job = {"stage": "Queued", "pct": 2, "done": False, "result": None}
        IMAGE_JOBS[jid] = job

        def progress(stage, pct):
            job["stage"] = stage
            job["pct"] = max(job["pct"], pct)

        def run():
            try:
                res = self.image_analyzer.analyze(data, fname, mode, spacing, progress=progress)
            except Exception as e:
                print(f"Image error: {e}")
                res = {"success": False, "error": str(e)}
            job["result"] = res
            job["pct"] = 100
            job["done"] = True

        threading.Thread(target=run, daemon=True).start()
        return jsonify({'success': True, 'job': jid})

    @self.app.route('/api/analyze_status/<jid>')
    def analyze_status(jid):
        job = IMAGE_JOBS.get(jid)
        if not job:
            return jsonify({'success': False, 'error': 'Unknown or expired job.'}), 404
        if job["done"]:
            IMAGE_JOBS.pop(jid, None)
            return jsonify({'success': True, 'done': True, 'stage': 'Done', 'pct': 100, 'result': job["result"]})
        return jsonify({'success': True, 'done': False, 'stage': job["stage"], 'pct': job["pct"]})


_fast_prev_init = WebUI.__init__
_fast_prev_get_html = WebUI.get_html


def _fast_init(self):
    _fast_prev_init(self)
    a = self.image_analyzer
    a._cache = {}
    a._cb = None
    a._job_lock = threading.Lock()
    if FLASK_AVAILABLE:
        _fast_add_routes(self)
    if os.getenv("PRELOAD_IMAGE_MODELS", "1") == "1":
        threading.Thread(target=a._preload, daemon=True).start()


# ---- 7) the animated "analyzing" screen ----
FAST_CSS = r"""
.an-wrap{display:flex;flex-direction:column;align-items:center;gap:14px;padding:8px 0}
.an-frame{position:relative;width:min(100%,380px);aspect-ratio:1/1;border-radius:18px;overflow:hidden;
  border:2px solid rgba(192,132,252,.8);box-shadow:0 0 28px rgba(168,85,247,.55);background:rgba(0,0,0,.35);
  animation:anGlow 2s ease-in-out infinite alternate}
.an-frame img{width:100%;height:100%;object-fit:contain;display:block;filter:saturate(1.1) brightness(.95)}
.an-ph{width:100%;height:100%;display:grid;place-items:center;font-size:90px;animation:anPulse 1.6s ease-in-out infinite}
.an-grid{position:absolute;inset:0;pointer-events:none;
  background-image:linear-gradient(rgba(255,255,255,.12) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.12) 1px,transparent 1px);
  background-size:28px 28px}
.an-scan{position:absolute;left:0;right:0;height:14%;top:0;pointer-events:none;
  background:linear-gradient(180deg,transparent,rgba(232,121,249,.55),rgba(255,255,255,.9) 92%,transparent);
  box-shadow:0 0 22px 6px rgba(232,121,249,.8);animation:anScan 2.4s ease-in-out infinite}
.an-c{position:absolute;width:26px;height:26px;border:3px solid #fff;pointer-events:none;animation:anPulse 1.4s ease-in-out infinite}
.an-c.tl{top:8px;left:8px;border-right:0;border-bottom:0;border-top-left-radius:8px}
.an-c.tr{top:8px;right:8px;border-left:0;border-bottom:0;border-top-right-radius:8px}
.an-c.bl{bottom:8px;left:8px;border-right:0;border-top:0;border-bottom-left-radius:8px}
.an-c.br{bottom:8px;right:8px;border-left:0;border-top:0;border-bottom-right-radius:8px}
@keyframes anScan{0%{top:-14%}50%{top:100%}100%{top:-14%}}
@keyframes anGlow{from{box-shadow:0 0 14px rgba(168,85,247,.4)}to{box-shadow:0 0 38px rgba(232,121,249,.9)}}
@keyframes anPulse{0%,100%{opacity:.55;transform:scale(.97)}50%{opacity:1;transform:scale(1.03)}}
.an-stage{font-weight:700;font-size:1.1em;display:flex;align-items:center;gap:8px;text-align:center}
.an-dots{display:inline-flex;gap:4px}
.an-dots i{width:7px;height:7px;border-radius:50%;background:currentColor;opacity:.3;animation:anDot 1.2s infinite}
.an-dots i:nth-child(2){animation-delay:.2s}.an-dots i:nth-child(3){animation-delay:.4s}
@keyframes anDot{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}
.an-bar{width:min(100%,380px);height:14px;border-radius:10px;background:rgba(255,255,255,.18);overflow:hidden;border:1px solid rgba(255,255,255,.3)}
.an-bar i{display:block;height:100%;width:0;border-radius:10px;
  background:linear-gradient(90deg,#8b5cf6,#e879f9,#ff9ad5,#8b5cf6);background-size:200% 100%;animation:anShim 1.2s linear infinite;
  box-shadow:0 0 12px rgba(232,121,249,.8)}
@keyframes anShim{to{background-position:-200% 0}}
.an-meta{width:min(100%,380px);display:flex;justify-content:space-between;font-size:.9em;opacity:.85}
.an-log{list-style:none;padding:0!important;margin:0;width:min(100%,380px);font-size:.92em}
.an-log li{display:flex;gap:8px;align-items:center;margin:3px 0!important;opacity:.9}
.an-log li.done{opacity:.6}
.an-spin{width:12px;height:12px;border:2px solid rgba(255,255,255,.35);border-top-color:#e879f9;border-radius:50%;animation:anSpin .8s linear infinite;flex:none}
@keyframes anSpin{to{transform:rotate(360deg)}}
.an-hint{font-size:.85em;opacity:.8;text-align:center;min-height:1.2em}
body.light .an-frame{border-color:#ff7aa2;box-shadow:0 0 28px rgba(255,122,162,.6)}
body.light .an-bar{background:rgba(255,255,255,.55);border-color:rgba(60,120,200,.4)}
"""

FAST_JS = r"""
(function () {
  const fileIn = document.getElementById('img-file');
  const out = document.getElementById('img-output');
  const btn = document.querySelector('#image .generate-button');
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  function loaderHTML(src) {
    return '<div class="an-wrap">' +
      '<div class="an-frame">' + (src ? '<img src="' + src + '">' : '<div class="an-ph">🧠</div>') +
      '<div class="an-grid"></div><div class="an-scan"></div>' +
      '<span class="an-c tl"></span><span class="an-c tr"></span><span class="an-c bl"></span><span class="an-c br"></span></div>' +
      '<div class="an-stage"><span id="an-stage">Uploading</span><span class="an-dots"><i></i><i></i><i></i></span></div>' +
      '<div class="an-bar"><i id="an-fill"></i></div>' +
      '<div class="an-meta"><span id="an-pct">0%</span><span id="an-time">0s</span></div>' +
      '<ul class="an-log" id="an-log"></ul>' +
      '<div class="an-hint" id="an-hint"></div></div>';
  }

  window.analyzeImage = async function () {
    const f = fileIn.files[0];
    if (!f) { out.textContent = 'Please choose an image first.'; return; }
    const prev = document.getElementById('img-preview');
    btn.disabled = true;
    out.innerHTML = loaderHTML(prev.style.display !== 'none' ? prev.src : '');
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    const $ = id => document.getElementById(id);
    let target = 3, shown = 0, done = false, lastStage = '';
    const t0 = Date.now();

    const timer = setInterval(() => {
      shown += (target - shown) * 0.1;                        // glide to the real value
      if (!done && shown < Math.min(target + 8, 94)) shown += 0.04;   // keep creeping so it never looks frozen
      $('an-fill').style.width = shown + '%';
      $('an-pct').textContent = Math.round(shown) + '%';
      const s = Math.floor((Date.now() - t0) / 1000);
      $('an-time').textContent = s + 's';
      if (s > 25) $('an-hint').textContent = 'Still working. The first run downloads the AI models, so it can take a few minutes. Next time it will be fast.';
    }, 100);

    function setStage(name) {
      if (name === lastStage) return;
      lastStage = name;
      $('an-stage').textContent = name;
      const log = $('an-log');
      log.querySelectorAll('li:not(.done)').forEach(li => { li.className = 'done'; li.firstChild.outerHTML = '<span>✅</span>'; });
      const li = document.createElement('li');
      li.innerHTML = '<span class="an-spin"></span><span>' + escapeHtml(name) + '</span>';
      log.appendChild(li);
    }

    try {
    async function readApiJson(response, endpoint) {
  const body = await response.text();

  try {
    return JSON.parse(body);
  } catch (e) {
    throw new Error(
      endpoint + ' returned HTML or invalid JSON. HTTP status: ' +
      response.status + '. Response: ' + body.slice(0, 200)
    );
  }
}
      const fd = new FormData();
      fd.append('image', f);
      fd.append('mode', $('img-mode') ? document.getElementById('img-mode').value : 'auto');
      fd.append('spacing', document.getElementById('img-spacing').value);
      const startResponse = await fetch('/api/analyze_start', {
  method: 'POST',
  body: fd
});

const start = await readApiJson(startResponse, '/api/analyze_start');
      if (!start.success) throw new Error(start.error || 'Could not start the analysis.');

      let result = null, fails = 0;
      while (!result) {
        await sleep(400);
        try {
          const statusResponse = await fetch('/api/analyze_status/' + start.job);

const s = await readApiJson(
  statusResponse,
  '/api/analyze_status/' + start.job
);
          fails = 0;
          if (!s.success) throw new Error(s.error);
          target = Math.max(target, s.pct);
          setStage(s.stage);
          if (s.done) result = s.result;
        } catch (e) {
          if (++fails > 15) throw e;
        }
      }
      done = true; target = 100;
      await sleep(450);
      clearInterval(timer);
      if (!result.success) { out.textContent = result.error || 'Something went wrong.'; }
      else { out.innerHTML = result.kind === 'mri' ? window.__renderMri(result) : window.__renderObjects(result); }
      out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
      clearInterval(timer);
      out.textContent = 'Error: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  };
})();
"""

# expose the old result renderers so the new loader can reuse them
IMG_JS = IMG_JS.replace("function renderMri(d) {", "window.__renderMri = renderMri;\n  function renderMri(d) {", 1)
IMG_JS = IMG_JS.replace("function renderObjects(d) {", "window.__renderObjects = renderObjects;\n  function renderObjects(d) {", 1)
IMG_JS = IMG_JS.replace("window.analyzeImage = async function () {", "window.__oldAnalyze = async function () {", 1)


def _fast_get_html(self) -> str:
    html = _fast_prev_get_html(self)
    html = html.replace('</head>', '<style>' + FAST_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + FAST_JS + '</script></body>', 1)
    return html


WebUI.__init__ = _fast_init
WebUI.get_html = _fast_get_html

# ============================================================================
# LLM VISION PATCH: Gemini looks at the image itself (open vocabulary),
# Hugging Face RT-DETR + BLIP-large are only the backup.
# The chatbot also remembers the image, so you can ask follow-ups in Chat.
# ============================================================================
import json as _json

# ---- A) Backup models (better than before) ----
ImageAnalyzer.OBJECT_MODEL = os.getenv("OBJECT_MODEL", "PekingU/rtdetr_r101vd_coco_o365")
ImageAnalyzer.CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")

VISION_PROMPT = """Look at this image carefully and work out by yourself what is in it.
Return ONLY a JSON object (no markdown) with exactly these keys:
{
  "scene": "short title of the scene, e.g. 'busy street market at night'",
  "description": "2-4 sentences describing what is happening",
  "objects": [
    {"label": "specific name, e.g. 'golden retriever', not just 'dog'",
     "details": "color, material, state, brand/model ONLY if clearly readable",
     "box_2d": [ymin, xmin, ymax, xmax],
     "uncertain": false}
  ],
  "people": ["short description of each person: approx. age group, clothing, what they do"],
  "text_in_image": ["any readable text, signs, labels"]
}
Rules:
- box_2d uses coordinates normalized to 0-1000 (top-left is 0,0).
- One entry per visible object instance, each with its own box. Include small objects too.
- Never invent things you cannot see. If you are not sure, set "uncertain": true.
- Use empty lists when there are no people or no text."""


def _parse_json_loose(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
    try:
        return _json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            return _json.loads(m.group(0))
        raise


def _to_jpeg(img, quality=90):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _jpeg_from_upload(data, filename):
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        return None
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((1600, 1600))
        return _to_jpeg(im)
    except Exception:
        return None


# ---- B) The chatbot gets eyes ----
SimpleChatbot.image_ctx = None


def _bot_vision_json(self, jpeg_bytes, prompt=VISION_PROMPT):
    if not self._ensure_client():
        raise RuntimeError(self.last_error or "Gemini is not available.")
    models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
    tried_discovery = False
    last = None
    while True:
        for m in models:
            try:
                resp = self.client.models.generate_content(
                    model=m,
                    contents=[genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"), prompt],
                    config=genai_types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                txt = (getattr(resp, "text", None) or "").strip()
                if txt:
                    self.chat_model = m
                    return txt
                last = f"{m}: empty answer"
            except Exception as e:
                last = f"{m}: {e}"
                self.last_error = last
        if tried_discovery:
            break
        tried_discovery = True
        found, _ = self._discover_models()
        models = [x for x in found if x not in models][:4]
        if not models:
            break
    raise RuntimeError(last or "Gemini could not read the image.")


def _bot_set_image_context(self, jpeg_bytes, summary):
    self.image_ctx = {"bytes": jpeg_bytes, "summary": summary}


def _bot_clear_history(self):
    self.history = []
    self.memory = []
    self.image_ctx = None


_orig_try_models = SimpleChatbot._try_models


def _try_models_with_image(self, models, contents, system_prompt):
    ctx = self.image_ctx
    if ctx:
        note = ("This is the image I uploaded in the Image Analysis tab. "
                "Use it to answer my questions about it." if ctx["bytes"] else
                "I analysed an image in the Image Analysis tab. Here is the result.")
        parts = []
        if ctx["bytes"]:
            parts.append(genai_types.Part.from_bytes(data=ctx["bytes"], mime_type="image/jpeg"))
        parts.append(genai_types.Part(text=note))
        prefix = [
            genai_types.Content(role="user", parts=parts),
            genai_types.Content(role="model", parts=[genai_types.Part(text=ctx["summary"])]),
        ]
        contents = prefix + list(contents)
    return _orig_try_models(self, models, contents, system_prompt)


SimpleChatbot.vision_json = _bot_vision_json
SimpleChatbot.set_image_context = _bot_set_image_context
SimpleChatbot.clear_history = _bot_clear_history
SimpleChatbot._try_models = _try_models_with_image


# ---- C) Image analyzer: Gemini first, Hugging Face as backup ----
_hf_objects = ImageAnalyzer._analyze_objects      # RT-DETR + BLIP (already has progress stage)


def _llm_objects(self, img):
    W, H = img.size
    bot = getattr(self, "chatbot", None)
    try:
        if bot is None:
            raise RuntimeError("Chatbot is not connected to the image analyzer.")
        data = _parse_json_loose(bot.vision_json(_to_jpeg(img)))
    except Exception as e:
        print("Gemini vision failed, using Hugging Face backup:", str(e)[:200])
        try:
            res = _hf_objects(self, img)
        except Exception:
            self.OBJECT_MODEL = "facebook/detr-resnet-101"
            self._pipes.pop("det", None)
            res = _hf_objects(self, img)
        res["source"] = "Hugging Face models (RT-DETR + BLIP). Gemini was not used."
        res["fallback_reason"] = str(e)[:200]
        return res

    draw_img = img.copy()
    d = ImageDraw.Draw(draw_img)
    font = self._font(max(13, int(min(W, H) / 38)))
    lw = max(2, int(min(W, H) / 250))
    objects, counts = [], {}

    for o in (data.get("objects") or []):
        if not isinstance(o, dict):
            continue
        label = str(o.get("label") or "object").strip()
        counts[label] = counts.get(label, 0) + 1
        entry = {"label": label, "details": str(o.get("details") or ""), "confidence": None,
                 "uncertain": bool(o.get("uncertain")), "width_px": None, "height_px": None,
                 "area_percent": None, "size_class": None, "position": None}
        box = o.get("box_2d")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                y0, x0, y1, x1 = [float(v) for v in box]
                X0, X1 = sorted((x0 / 1000 * W, x1 / 1000 * W))
                Y0, Y1 = sorted((y0 / 1000 * H, y1 / 1000 * H))
                X0, Y0 = max(0, X0), max(0, Y0)
                X1, Y1 = min(W - 1, X1), min(H - 1, Y1)
                w, h = X1 - X0, Y1 - Y0
                if w > 2 and h > 2:
                    pct = w * h / (W * H) * 100
                    entry.update({"width_px": int(w), "height_px": int(h),
                                  "area_percent": round(pct, 2), "size_class": self._size_class(pct),
                                  "position": self._position((X0 + X1) / 2, (Y0 + Y1) / 2, W, H)})
                    col = self._color(label)
                    d.rectangle([X0, Y0, X1, Y1], outline=col, width=lw)
                    text = label + (" ?" if entry["uncertain"] else "")
                    tb = d.textbbox((0, 0), text, font=font)
                    th = tb[3] - tb[1] + 6
                    ty = Y0 - th if Y0 - th > 0 else Y0
                    d.rectangle([X0, ty, X0 + tb[2] - tb[0] + 8, ty + th], fill=col)
                    d.text((X0 + 4, ty + 2), text, fill=(255, 255, 255), font=font)
            except (TypeError, ValueError):
                pass
        objects.append(entry)

    return {
        "kind": "objects",
        "source": "Gemini vision (the chatbot's LLM looked at the image itself)",
        "scene": data.get("scene") or "",
        "caption": data.get("description") or "",
        "objects": objects,
        "object_counts": counts,
        "people": [str(p) for p in (data.get("people") or [])],
        "text_in_image": [str(t) for t in (data.get("text_in_image") or [])],
        "annotated_image": self._to_data_url(draw_img),
    }


ImageAnalyzer._analyze_objects = _stage("Asking the AI to look at the image", 40)(_llm_objects)


# Preload only what is always needed; the backup models load only if Gemini fails.
def _preload2(self):
    with self._lock:
        try:
            self._pipe("clip", "zero-shot-image-classification", self.CLIP_MODEL)
            self._load_mri_models()
            print("✓ Image models preloaded")
        except Exception as e:
            print("Image model preload problem:", str(e)[:150])


ImageAnalyzer._preload = _preload2


# ---- D) After every analysis, hand the image + result to the chatbot ----
def _summarize(res):
    if res.get("kind") == "mri":
        t = res.get("tumor") or {}
        s = "Brain MRI analysis: " + str(t.get("type", "unknown")) + f" ({t.get('confidence', '?')}% confidence)."
        if t.get("location"):
            s += " Location: " + t["location"] + "."
        if t.get("size"):
            s += " Size: " + json.dumps(t["size"]) + "."
        return s + " This is an AI screening aid, not a diagnosis."
    parts = []
    if res.get("scene"):
        parts.append("Scene: " + res["scene"] + ".")
    if res.get("caption"):
        parts.append(res["caption"])
    objs = [o["label"] + (f" ({o['details']})" if o.get("details") else "") for o in res.get("objects", [])]
    if objs:
        parts.append("Objects I found: " + "; ".join(objs) + ".")
    if res.get("people"):
        parts.append("People: " + "; ".join(res["people"]) + ".")
    if res.get("text_in_image"):
        parts.append("Text in the image: " + "; ".join(res["text_in_image"]) + ".")
    return " ".join(parts) or "I analysed the image."


_share_prev_analyze = ImageAnalyzer.analyze


def _analyze_and_share(self, data, filename="", mode="auto", spacing=None, progress=None):
    res = _share_prev_analyze(self, data, filename, mode, spacing, progress)
    bot = getattr(self, "chatbot", None)
    if bot is not None and res.get("success"):
        try:
            bot.set_image_context(_jpeg_from_upload(data, filename), _summarize(res))
        except Exception as e:
            print("Could not share image with chatbot:", e)
    return res


ImageAnalyzer.analyze = _analyze_and_share

_llm_prev_init = WebUI.__init__


def _llm_init(self):
    _llm_prev_init(self)
    self.image_analyzer.chatbot = self.chatbot


WebUI.__init__ = _llm_init

# ---- E) New result screen for ordinary images ----
LLM_CSS = r"""
.img-src{display:inline-block;margin:0 0 10px;padding:4px 12px;border-radius:20px;font-size:.85em;
  background:rgba(168,85,247,.25);border:1px solid rgba(192,132,252,.6)}
.img-tip{margin-top:14px;padding:10px 14px;border-radius:12px;background:rgba(60,200,120,.15);border:1px solid rgba(60,200,120,.5)}
#img-output small{opacity:.75}
"""

LLM_JS = r"""
(function () {
  const esc = s => escapeHtml(String(s));
  const title = t => '<div class="cr-title">' + t + '</div>';
  const list = a => '<ul>' + a.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul>';
  const dash = v => (v === null || v === undefined || v === '') ? '—' : v;

  window.__renderObjects = function (d) {
    let h = '<div class="img-src">🤖 ' + esc(d.source || '') + '</div>';
    if (d.fallback_reason) h += '<div class="img-disc">Gemini could not be used (' + esc(d.fallback_reason) + '), so the backup models answered.</div>';
    if (d.scene) h += title('🎬 Scene') + '<p>' + esc(d.scene) + '</p>';
    if (d.caption) h += title('👁️ What I see') + '<p>' + esc(d.caption) + '</p>';
    h += '<img class="result" src="' + d.annotated_image + '">';
    const c = Object.entries(d.object_counts || {});
    if (c.length) h += title('📦 Objects found') + '<p>' + c.map(([k, v]) => v + ' × ' + esc(k)).join(', ') + '</p>';
    if (d.objects && d.objects.length) {
      h += '<table><tr><th>Object</th><th>Details</th><th>Size (px)</th><th>% of image</th><th>Position</th></tr>' +
        d.objects.map(o => '<tr><td>' + esc(o.label) + (o.uncertain ? ' <small>(unsure)</small>' : '') + '</td><td>' + esc(dash(o.details)) +
          '</td><td>' + (o.width_px ? o.width_px + '×' + o.height_px : '—') + '</td><td>' + dash(o.area_percent !== null && o.area_percent !== undefined ? o.area_percent + '%' : null) +
          '</td><td>' + dash(o.position) + '</td></tr>').join('') + '</table>';
    } else {
      h += '<p>No objects detected.</p>';
    }
    if (d.people && d.people.length) h += title('🧍 People') + list(d.people);
    if (d.text_in_image && d.text_in_image.length) h += title('🔤 Text in the image') + list(d.text_in_image);
    h += '<div class="img-tip">💬 Want to know more? Open the <b>Chat</b> tab and ask me anything about this image.</div>';
    return h;
  };
})();
"""

_llm_prev_get_html = WebUI.get_html


def _llm_get_html(self) -> str:
    html = _llm_prev_get_html(self)
    html = html.replace('</head>', '<style>' + LLM_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + LLM_JS + '</script></body>', 1)
    return html


WebUI.get_html = _llm_get_html

# ============================================================================
# PEOPLE DETECTION PATCH: man / woman / boy / girl / child / baby
# Gemini does it first; RT-DETR + CLIP are the backup.
# ============================================================================

PEOPLE_PROMPT = """Look at this image carefully and work out by yourself what is in it.
Return ONLY a JSON object (no markdown) with exactly these keys:
{
  "scene": "short title of the scene",
  "description": "2-4 sentences describing what is happening",
  "objects": [
    {"label": "specific name, e.g. 'golden retriever', not just 'dog'",
     "details": "color, material, state, brand/model ONLY if clearly readable",
     "box_2d": [ymin, xmin, ymax, xmax],
     "uncertain": false}
  ],
  "persons": [
    {"category": "man | woman | boy | girl | child | baby | person",
     "age_group": "baby | child | teen | adult | senior",
     "description": "clothing, hair, what they are doing",
     "box_2d": [ymin, xmin, ymax, xmax],
     "uncertain": false}
  ],
  "text_in_image": ["any readable text, signs, labels"]
}
Rules:
- box_2d uses coordinates normalized to 0-1000 (top-left is 0,0).
- EVERY visible human goes in "persons", one entry per person, each with its own box. Include people in the background and partly visible people. Do NOT put people in "objects".
- Use "man" or "woman" for adults and "boy" or "girl" for children. Use "child" if the child's gender is not clear, and "baby" for infants.
- Decide only from how the person looks in the picture. If you are not sure, use "person" or set "uncertain": true.
- Never name a real person and never guess who someone is.
- One entry per visible object instance, each with its own box. Include small objects too.
- Never invent things you cannot see.
- Use empty lists when there are no persons, objects or text."""

PERSON_PLURALS = {"man": ("man", "men"), "woman": ("woman", "women"), "boy": ("boy", "boys"),
                  "girl": ("girl", "girls"), "child": ("child", "children"),
                  "baby": ("baby", "babies"), "person": ("person", "people")}
PERSON_ORDER = ["man", "woman", "boy", "girl", "child", "baby", "person"]
PERSON_COLORS = {"man": (37, 99, 235), "woman": (219, 39, 119), "boy": (22, 163, 74),
                 "girl": (234, 88, 12), "child": (13, 148, 136), "baby": (202, 138, 4),
                 "person": (100, 116, 139)}

# CLIP labels for the backup path -> (group, age group)
HF_PERSON_LABELS = {
    "a photo of an adult man": ("man", "adult"),
    "a photo of an adult woman": ("woman", "adult"),
    "a photo of an elderly man": ("man", "senior"),
    "a photo of an elderly woman": ("woman", "senior"),
    "a photo of a boy": ("boy", "child"),
    "a photo of a girl": ("girl", "child"),
    "a photo of a baby": ("baby", "baby"),
}


def _person_group(cat):
    c = str(cat or "").lower()
    if any(k in c for k in ("baby", "infant", "toddler")):
        return "baby"
    if "boy" in c:
        return "boy"
    if "girl" in c:
        return "girl"
    if any(k in c for k in ("child", "kid")):
        return "child"
    if any(k in c for k in ("woman", "female", "lady")):   # must be checked before "man"
        return "woman"
    if any(k in c for k in ("man", "male", "gentleman")):
        return "man"
    return "person"


def _people_summary(persons):
    if not persons:
        return "No people found"
    cnt = {}
    for p in persons:
        cnt[p["group"]] = cnt.get(p["group"], 0) + 1
    parts = []
    for g in PERSON_ORDER:
        n = cnt.get(g, 0)
        if n:
            one, many = PERSON_PLURALS[g]
            parts.append(f"{n} {one if n == 1 else many}")
    total = len(persons)
    return f"{total} {'person' if total == 1 else 'people'}: " + ", ".join(parts)


def _norm_box(box, W, H):
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        y0, x0, y1, x1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    X0, X1 = sorted((x0 / 1000 * W, x1 / 1000 * W))
    Y0, Y1 = sorted((y0 / 1000 * H, y1 / 1000 * H))
    X0, Y0 = max(0, X0), max(0, Y0)
    X1, Y1 = min(W - 1, X1), min(H - 1, Y1)
    return (X0, Y0, X1, Y1) if (X1 - X0 > 2 and Y1 - Y0 > 2) else None


def _box_meta(self, box, W, H):
    X0, Y0, X1, Y1 = box
    w, h = X1 - X0, Y1 - Y0
    pct = w * h / (W * H) * 100
    return {"width_px": int(w), "height_px": int(h), "area_percent": round(pct, 2),
            "size_class": self._size_class(pct),
            "position": self._position((X0 + X1) / 2, (Y0 + Y1) / 2, W, H)}


def _draw_persons(self, img, persons):
    W, H = img.size
    d = ImageDraw.Draw(img)
    font = self._font(max(13, int(min(W, H) / 36)))
    lw = max(3, int(min(W, H) / 170))
    for p in persons:
        if "_box" not in p:
            continue
        x0, y0, x1, y1 = p["_box"]
        col = PERSON_COLORS.get(p["group"], PERSON_COLORS["person"])
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        text = p["label"] + (" ?" if p.get("uncertain") else "")
        tb = d.textbbox((0, 0), text, font=font)
        th = tb[3] - tb[1] + 6
        ty = y0 - th if y0 - th > 0 else y0
        d.rectangle([x0, ty, x0 + tb[2] - tb[0] + 8, ty + th], fill=col)
        d.text((x0 + 4, ty + 2), text, fill=(255, 255, 255), font=font)


def _finish_persons(persons):
    return [{k: v for k, v in p.items() if k != "_box"} for p in persons]


# ---------------- backup: RT-DETR finds people, CLIP says man/woman/kid ----------------
def _hf_people(self, img):
    W, H = img.size
    det = self._pipe("det", "object-detection", self.OBJECT_MODEL)
    clip = self._pipe("clip", "zero-shot-image-classification", self.CLIP_MODEL)
    boxes = [o for o in det(img) if o["label"] == "person" and o["score"] >= 0.5]
    boxes.sort(key=lambda o: o["box"]["xmin"])
    persons, counters = [], {}
    for o in boxes:
        b = o["box"]
        X0, Y0, X1, Y1 = max(0, b["xmin"]), max(0, b["ymin"]), min(W - 1, b["xmax"]), min(H - 1, b["ymax"])
        if X1 - X0 < 24 or Y1 - Y0 < 24:
            continue
        pad = 0.05
        crop = img.crop((max(0, X0 - (X1 - X0) * pad), max(0, Y0 - (Y1 - Y0) * pad),
                         min(W, X1 + (X1 - X0) * pad), min(H, Y1 + (Y1 - Y0) * pad)))
        try:
            top = clip(crop, candidate_labels=list(HF_PERSON_LABELS))[0]
            group, age = HF_PERSON_LABELS[top["label"]]
            unsure = top["score"] < 0.5
        except Exception:
            group, age, unsure = "person", "", True
        counters[group] = counters.get(group, 0) + 1
        entry = {"label": f"{group.title()} {counters[group]}", "group": group, "age_group": age,
                 "description": "", "uncertain": unsure, "_box": (X0, Y0, X1, Y1)}
        entry.update(_box_meta(self, (X0, Y0, X1, Y1), W, H))
        persons.append(entry)
    return persons


def _people_fallback(self, img, err):
    try:
        res = _hf_objects(self, img)
    except Exception:
        self.OBJECT_MODEL = "facebook/detr-resnet-101"
        self._pipes.pop("det", None)
        res = _hf_objects(self, img)
    res["source"] = "Hugging Face models (RT-DETR + BLIP + CLIP). Gemini was not used."
    res["fallback_reason"] = str(err)[:200]
    persons = []
    try:
        persons = _hf_people(self, img)
    except Exception as e2:
        print("Backup people detection failed:", str(e2)[:150])
    res["objects"] = [o for o in res.get("objects", []) if o["label"] != "person"]
    res["object_counts"] = {k: v for k, v in res.get("object_counts", {}).items() if k != "person"}
    if persons:
        raw = base64.b64decode(res["annotated_image"].split(",", 1)[1])
        base_img = Image.open(io.BytesIO(raw)).convert("RGB")
        _draw_persons(self, base_img, persons)
        res["annotated_image"] = self._to_data_url(base_img)
    persons = _finish_persons(persons)
    res["persons"] = persons
    res["people_summary"] = _people_summary(persons)
    res["people"] = [f"{p['label']} ({p['age_group']})" if p["age_group"] else p["label"] for p in persons]
    return res


# ---------------- main path: Gemini ----------------
def _llm_objects2(self, img):
    W, H = img.size
    bot = getattr(self, "chatbot", None)
    try:
        if bot is None:
            raise RuntimeError("Chatbot is not connected to the image analyzer.")
        data = _parse_json_loose(bot.vision_json(_to_jpeg(img), PEOPLE_PROMPT))
    except Exception as e:
        print("Gemini vision failed, using Hugging Face backup:", str(e)[:200])
        return _people_fallback(self, img, e)

    draw_img = img.copy()
    d = ImageDraw.Draw(draw_img)
    font = self._font(max(13, int(min(W, H) / 38)))
    lw = max(2, int(min(W, H) / 250))

    # ----- objects -----
    objects, counts = [], {}
    for o in (data.get("objects") or []):
        if not isinstance(o, dict):
            continue
        label = str(o.get("label") or "object").strip()
        counts[label] = counts.get(label, 0) + 1
        entry = {"label": label, "details": str(o.get("details") or ""), "confidence": None,
                 "uncertain": bool(o.get("uncertain")), "width_px": None, "height_px": None,
                 "area_percent": None, "size_class": None, "position": None}
        box = _norm_box(o.get("box_2d"), W, H)
        if box:
            entry.update(_box_meta(self, box, W, H))
            X0, Y0, X1, Y1 = box
            col = self._color(label)
            d.rectangle([X0, Y0, X1, Y1], outline=col, width=lw)
            text = label + (" ?" if entry["uncertain"] else "")
            tb = d.textbbox((0, 0), text, font=font)
            th = tb[3] - tb[1] + 6
            ty = Y0 - th if Y0 - th > 0 else Y0
            d.rectangle([X0, ty, X0 + tb[2] - tb[0] + 8, ty + th], fill=col)
            d.text((X0 + 4, ty + 2), text, fill=(255, 255, 255), font=font)
        objects.append(entry)

    # ----- people -----
    persons, counters = [], {}
    for p in (data.get("persons") or []):
        if not isinstance(p, dict):
            continue
        group = _person_group(p.get("category"))
        counters[group] = counters.get(group, 0) + 1
        entry = {"label": f"{group.title()} {counters[group]}", "group": group,
                 "age_group": str(p.get("age_group") or ""),
                 "description": str(p.get("description") or ""),
                 "uncertain": bool(p.get("uncertain")),
                 "width_px": None, "height_px": None, "area_percent": None,
                 "size_class": None, "position": None}
        box = _norm_box(p.get("box_2d"), W, H)
        if box:
            entry.update(_box_meta(self, box, W, H))
            entry["_box"] = box
        persons.append(entry)
    _draw_persons(self, draw_img, persons)
    persons = _finish_persons(persons)

    return {
        "kind": "objects",
        "source": "Gemini vision (the chatbot's LLM looked at the image itself)",
        "scene": data.get("scene") or "",
        "caption": data.get("description") or "",
        "objects": objects,
        "object_counts": counts,
        "persons": persons,
        "people_summary": _people_summary(persons),
        "people": [f"{p['label']}" + (f" ({p['age_group']})" if p["age_group"] else "") +
                   (": " + p["description"] if p["description"] else "") for p in persons],
        "text_in_image": [str(t) for t in (data.get("text_in_image") or [])],
        "annotated_image": self._to_data_url(draw_img),
    }


ImageAnalyzer._analyze_objects = _stage("Asking the AI to look at the image", 40)(_llm_objects2)


# ---------------- tell the chatbot how many men / women / kids there are ----------------
_prev_summarize = _summarize


def _summarize(res):
    s = _prev_summarize(res)
    if res.get("kind") != "mri" and res.get("people_summary"):
        s = "People count: " + res["people_summary"] + ". " + s
    return s


# ---------------- result screen ----------------
PPL_CSS = r"""
.ppl-chips{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 12px}
.ppl-chip{padding:8px 16px;border-radius:999px;font-weight:700;background:rgba(168,85,247,.22);
  border:1px solid rgba(192,132,252,.7)}
.ppl-note{font-size:.82em;opacity:.75;margin-top:6px}
body.light .ppl-chip{background:rgba(255,255,255,.7);border-color:#ff7aa2}
"""

PPL_JS = r"""
(function () {
  const esc = s => escapeHtml(String(s));
  const ICON = {man: '👨', woman: '👩', boy: '👦', girl: '👧', child: '🧒', baby: '👶', person: '🧑'};
  const NAMES = {man: ['man', 'men'], woman: ['woman', 'women'], boy: ['boy', 'boys'], girl: ['girl', 'girls'],
                 child: ['child', 'children'], baby: ['baby', 'babies'], person: ['person', 'people']};
  const ORDER = ['man', 'woman', 'boy', 'girl', 'child', 'baby', 'person'];
  const prev = window.__renderObjects;

  window.__renderObjects = function (d) {
    let h = prev(Object.assign({}, d, { people: [] }));   // hide the old plain people list
    const ps = d.persons || [];
    let block = '<div class="cr-title">👥 People</div>';
    if (!ps.length) {
      block += '<p>No people found in this image.</p>';
    } else {
      const cnt = {};
      ps.forEach(p => { cnt[p.group] = (cnt[p.group] || 0) + 1; });
      block += '<div class="ppl-chips">' + ORDER.filter(g => cnt[g]).map(g =>
        '<span class="ppl-chip">' + ICON[g] + ' ' + cnt[g] + ' ' + NAMES[g][cnt[g] === 1 ? 0 : 1] + '</span>').join('') + '</div>';
      block += '<table><tr><th>Who</th><th>Age group</th><th>Description</th><th>Size (px)</th><th>Position</th></tr>' +
        ps.map(p => '<tr><td>' + ICON[p.group] + ' ' + esc(p.label) + (p.uncertain ? ' <small>(unsure)</small>' : '') +
          '</td><td>' + esc(p.age_group || '—') + '</td><td>' + esc(p.description || '—') +
          '</td><td>' + (p.width_px ? p.width_px + '×' + p.height_px : '—') +
          '</td><td>' + esc(p.position || '—') + '</td></tr>').join('') + '</table>';
      block += '<div class="ppl-note">Man, woman and kid are guessed from how people look in the picture, so they can be wrong. I never identify who a person is.</div>';
    }
    return h.replace('<div class="img-tip">', () => block + '<div class="img-tip">');
  };
})();
"""

_ppl_prev_get_html = WebUI.get_html


def _ppl_get_html(self) -> str:
    html = _ppl_prev_get_html(self)
    html = html.replace('</head>', '<style>' + PPL_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + PPL_JS + '</script></body>', 1)
    return html


WebUI.get_html = _ppl_get_html

# ============================================================================
# STRONG FAKE NEWS PATCH
# Gemini + Google Search fact-checks the claims. Domain checks, writing-style
# rules and a Hugging Face model are extra signals. You can paste a link too.
# ============================================================================
import datetime as _dt
import socket
import ipaddress
from urllib.parse import urlparse, urljoin

FN_TRUSTED = [
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'nytimes.com', 'washingtonpost.com',
    'theguardian.com', 'wsj.com', 'bloomberg.com', 'economist.com', 'ft.com', 'npr.org', 'pbs.org',
    'aljazeera.com', 'dw.com', 'france24.com', 'cnn.com', 'nbcnews.com', 'cbsnews.com',
    'abcnews.go.com', 'usatoday.com', 'time.com', 'politico.com', 'axios.com',
    'nature.com', 'science.org', 'pnas.org', 'thelancet.com', 'nejm.org', 'bmj.com',
    'scientificamerican.com', 'cdc.gov', 'who.int', 'nasa.gov', 'nih.gov', 'fda.gov', 'un.org',
    'europa.eu', 'worldbank.org', 'imf.org', 'snopes.com', 'factcheck.org', 'politifact.com',
    'fullfact.org',
]
FN_SATIRE = ['theonion.com', 'babylonbee.com', 'clickhole.com', 'waterfordwhispersnews.com',
             'thedailymash.co.uk', 'newsthump.com']
FN_SHORT = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'cutt.ly', 'rebrand.ly']
FN_SKETCHY_TLD = ('.xyz', '.click', '.top', '.buzz', '.icu', '.work', '.rest', '.gq', '.tk')
_FN_GENERIC = {'nature', 'science', 'time', 'politico', 'axios', 'economist'}
_FN_LABELS = {d.split('.')[0] for d in FN_TRUSTED} - _FN_GENERIC
_FN_LABELS = {l for l in _FN_LABELS if len(l) >= 5 or l in ('bbc', 'wsj', 'cnn', 'npr')}
_FN_URL = r"https?://[^\s<>\"')\]]+"


def _fn_host(url):
    try:
        h = (urlparse(url).hostname or '').lower().strip('.')
    except Exception:
        return ''
    return h[4:] if h.startswith('www.') else h


def _fn_reg(host):
    p = host.split('.')
    if len(p) >= 3 and p[-2] in ('co', 'com', 'org', 'gov', 'ac', 'net') and len(p[-1]) == 2:
        return '.'.join(p[-3:])
    return '.'.join(p[-2:])


def _fn_match(host, domains):
    return any(host == d or host.endswith('.' + d) for d in domains)


def _fn_is_trusted(host):
    return _fn_match(host, FN_TRUSTED) or host.endswith(('.gov', '.edu'))


def _fn_analyze_domains(hosts):
    a = {'trusted_count': 0, 'untrusted_count': 0, 'no_sources': not hosts,
         'has_trusted_sources': False, 'lookalike': [], 'spoof_hosts': [],
         'satire': [], 'shortened': [], 'sketchy_tld': [], 'flags': []}
    for h in hosts:
        if _fn_match(h, FN_SATIRE):
            a['satire'].append(h)
            continue
        if _fn_match(h, FN_SHORT):
            a['shortened'].append(h)
            a['untrusted_count'] += 1
            a['flags'].append(f"Shortened link hides its real destination: {h}")
            continue
        if _fn_is_trusted(h):
            a['trusted_count'] += 1
            continue
        a['untrusted_count'] += 1
        tokens = re.split(r"[.\-_]", h)
        hit = [t for t in tokens if t in _FN_LABELS]
        close = difflib.get_close_matches(_fn_reg(h), FN_TRUSTED, n=1, cutoff=0.82)
        if hit or close:
            what = close[0] if close else hit[0]
            a['lookalike'].append(f"{h} (imitates {what})")
            a['spoof_hosts'].append(h)
            a['flags'].append(f"Link imitates a trusted outlet: {h} (looks like {what})")
        elif h.endswith(FN_SKETCHY_TLD):
            a['sketchy_tld'].append(h)
            a['flags'].append(f"Link uses a domain ending often used by spam sites: {h}")
    a['has_trusted_sources'] = a['trusted_count'] > 0
    return a


# ---------------- writing-style rules ----------------
_FN_RULES = [
    ("Sensationalist or clickbait wording", 10,
     r"\b(shocking|unbelievable|jaw[- ]dropping|you won'?t believe|this will shock you|must[- ]see|mind[- ]blowing|bombshell)\b"),
    ("Claims that the truth is being hidden from you", 14,
     r"they don'?t want you to know|the truth (they|the media)|mainstream media (won'?t|refuses|is hiding|is lying)|what (the )?(media|doctors|government) (isn'?t|won'?t|doesn'?t) (telling|tell)|wake up,? (people|america)|sheeple|big pharma|(being )?(censored|silenced|suppressed) (by|because)"),
    ("Pressure to share before it is 'deleted'", 12,
     r"share (this )?(before|now|everywhere|with everyone)|before (it'?s|they) (delete|remove|ban)|forward this|spread the word|going viral|share if you"),
    ("Miracle-cure or too-good-to-be-true health claims", 12,
     r"\b(miracle cure|cures? (cancer|diabetes|covid)|doctors hate|one weird trick)\b|\bdetox\b|lose \d+ (lbs|pounds|kg) in"),
    ("Absolute or unprovable certainty", 8,
     r"100% (proven|safe|true|guaranteed)|proven fact|undeniable proof|everyone knows|case closed|scientists are baffled|nobody is talking about"),
    ("Vague or anonymous sourcing", 8,
     r"\b(anonymous|unnamed) (source|official|insider)s?\b|\bsources? (say|said|claim)\b|\binsiders? (say|said|claim)\b|\bpeople are saying\b|\bmany (people )?(say|believe|think)\b|\bsome (say|believe)\b"),
    ("Artificial urgency", 5,
     r"\b(urgent|breaking|happening now|act now|before it'?s too late|immediate action required|don'?t ignore)\b"),
    ("Conspiracy-theory wording", 12,
     r"\b(deep state|false flag|crisis actors?|hoax|new world order|plandemic|chemtrails?|illuminati|globalists?)\b|cover[- ]?up|conspiracy"),
]
_FN_OUTRAGE = r"\b(outrage|outrageous|disgusting|evil|traitors?|scum|corrupt|sickening|vile)\b"
_FN_ATTR = (r"\b(according to|said|says|told|reported|announced|stated|confirmed|spokes(man|woman|person)|"
            r"study|published in|researchers?|press release|statement|officials?)\b")


def _fn_style(text):
    pats, pen = [], 0
    low = text.lower().replace("’", "'")
    for label, w, rx in _FN_RULES:
        if re.search(rx, low):
            pats.append(label)
            pen += w
    if len(re.findall(_FN_OUTRAGE, low)) >= 3:
        pats.append("Heavy outrage / emotional language")
        pen += 6
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 40:
        up = sum(1 for c in letters if c.isupper()) / len(letters)
        if up > 0.25:
            pats.append("Excessive use of capital letters")
            pen += 10
        elif len(re.findall(r"\b[A-Z]{4,}\b", text)) >= 5:
            pats.append("Many ALL-CAPS words")
            pen += 6
    if text.count('!') >= 3 or '!!' in text or '?!' in text:
        pats.append("Excessive exclamation marks")
        pen += 8
    if len(text) >= 400 and not re.search(_FN_ATTR, low):
        pats.append("No named people, organisations or studies are cited")
        pen += 12
    if len(text) < 80:
        pats.append("Very little text, so the analysis is limited")
        pen += 10
    return pats, max(0, 100 - pen)


# ---------------- Hugging Face text classifier (small extra signal) ----------------
def _fn_ml(self, text):
    if not TRANSFORMERS_AVAILABLE or getattr(self, '_fn_ml_failed', False):
        return None
    try:
        if getattr(self, '_fn_ml_pipe', None) is None:
            name = os.getenv('FAKE_NEWS_MODEL', 'jy46604790/Fake-News-Bert-Detect')
            print(f"Loading fake news model {name} ...")
            self._fn_ml_pipe = pipeline('text-classification', model=name, device=self.device)
        reals = []
        for i in range(0, min(len(text), 5400), 1800):
            out = self._fn_ml_pipe(text[i:i + 1800], truncation=True, max_length=512, top_k=None)
            if out and isinstance(out[0], list):
                out = out[0]
            fake = real = 0.0
            for d in out:
                lab = str(d['label']).lower()
                if 'fake' in lab or 'false' in lab or lab in ('label_0', '0'):
                    fake += float(d['score'])
                elif 'real' in lab or 'true' in lab or lab in ('label_1', '1'):
                    real += float(d['score'])
            if fake + real > 0:
                reals.append(real / (fake + real))
        return int(round(100 * sum(reals) / len(reals))) if reals else None
    except Exception as e:
        print("Fake news model problem:", str(e)[:150])
        self._fn_ml_failed = True
        return None


# ---------------- open a link safely ----------------
def _fn_public_ok(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for i in infos:
        ip = ipaddress.ip_address(i[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _fn_fetch(url):
    import requests
    import html as _html
    cur = url
    for _ in range(4):
        p = urlparse(cur)
        if p.scheme not in ('http', 'https') or not p.hostname or not _fn_public_ok(p.hostname):
            raise ValueError("That link can't be opened here. Paste the article text instead.")
        try:
            r = requests.get(cur, timeout=12, allow_redirects=False, stream=True,
                             headers={'User-Agent': 'Mozilla/5.0 (compatible; pip-ai-agent/1.0)'})
        except Exception as e:
            raise ValueError(f"Could not open the link ({str(e)[:80]}). Paste the article text instead.")
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get('Location')
            r.close()
            if not loc:
                break
            cur = urljoin(cur, loc)
            continue
        if r.status_code >= 400:
            r.close()
            raise ValueError(f"The website answered with error {r.status_code}. Paste the article text instead.")
        if 'html' not in r.headers.get('Content-Type', 'html').lower() and 'text' not in r.headers.get('Content-Type', '').lower():
            r.close()
            raise ValueError("That link is not a web page. Paste the article text instead.")
        raw = r.raw.read(1_500_000, decode_content=True)
        r.close()
        page = raw.decode(r.encoding or 'utf-8', errors='ignore')
        tm = re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S)
        title = _html.unescape(re.sub(r"\s+", " ", tm.group(1))).strip() if tm else ''
        page = re.sub(r"<(script|style|noscript|svg|nav|footer|header|form)\b.*?</\1>", " ", page, flags=re.I | re.S)
        am = re.search(r"<article\b.*?</article>", page, re.I | re.S)
        if am:
            page = am.group(0)
        body = _html.unescape(re.sub(r"<[^>]+>", " ", page))
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < 200:
            raise ValueError("I could not read enough text from that page (it may block bots or need JavaScript). Paste the article text instead.")
        return title, body[:8000]
    raise ValueError("Too many redirects. Paste the article text instead.")


# ---------------- Gemini + Google Search fact-check ----------------
FACT_PROMPT = """You are a careful professional fact-checker. Today's date is __TODAY__.
Check the article below. Use Google Search to verify its key claims against reliable, independent sources (news agencies, official bodies, scientific journals, fact-checking organisations).

Steps:
1. Pick the most important checkable factual claims (max 6). Ignore opinions.
2. Search for each claim and decide a verdict.
3. Decide whether it is satire, opinion or a real news report, and whether it uses manipulation techniques (emotional pressure, cherry-picking, fake urgency, missing context, invented quotes).

Return ONLY a JSON object (no markdown) with exactly these keys:
{
 "overall": "credible | mostly_credible | mixed | misleading | false | unverifiable | satire",
 "confidence": 0-100,
 "summary": "2-3 plain sentences: what is true and what is not",
 "article_type": "news report | opinion | satire | advertisement | social post | other",
 "claims": [{"claim": "...", "verdict": "true | mostly_true | misleading | false | unverifiable", "evidence": "1-2 sentences about what the sources say"}],
 "manipulation": ["short description of each technique found"],
 "missing_context": "important context the article leaves out, or empty string"
}
Rules:
- Base verdicts on what the search results say, not on how convincing the text sounds.
- If you find no reliable coverage at all, use "unverifiable". Never call something false only because you cannot find it.
- Keep the JSON keys and the verdict words in English. Write summary and evidence in the same language as the article.

ARTICLE__SRC__:
\"\"\"
__TEXT__
\"\"\""""


def _grounding_sources(resp):
    out, seen = [], set()
    try:
        gm = resp.candidates[0].grounding_metadata
        for ch in (getattr(gm, 'grounding_chunks', None) or []):
            w = getattr(ch, 'web', None)
            url = getattr(w, 'uri', None) if w else None
            if url and url not in seen:
                seen.add(url)
                out.append({'title': (getattr(w, 'title', None) or url)[:120], 'url': url})
    except Exception:
        pass
    return out


def _bot_fact_check(self, text, url=None):
    if not self._ensure_client():
        raise RuntimeError(self.last_error or "Gemini is not available.")
    prompt = (FACT_PROMPT.replace("__TODAY__", _dt.date.today().isoformat())
              .replace("__SRC__", f" (from {url})" if url else "")
              .replace("__TEXT__", text))
    models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
    last, discovered = None, False
    while True:
        for m in models:
            for with_search in (True, False):
                try:
                    cfg = genai_types.GenerateContentConfig(
                        temperature=0.1, max_output_tokens=6000,
                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())] if with_search else None)
                    resp = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                    txt = (getattr(resp, 'text', None) or '').strip()
                    if not txt:
                        last = f"{m}: empty answer"
                        continue
                    data = _parse_json_loose(txt)
                    self.chat_model = m
                    return data, _grounding_sources(resp), with_search
                except Exception as e:
                    last = f"{m}: {e}"
                    self.last_error = last
        if discovered:
            break
        discovered = True
        found, _ = self._discover_models()
        models = [x for x in found if x not in models][:4]
        if not models:
            break
    raise RuntimeError(last or "Gemini could not fact-check this.")


SimpleChatbot.fact_check = _bot_fact_check

CLAIM_PTS = {'true': 100, 'mostly_true': 78, 'misleading': 28, 'false': 0}
OVERALL_PTS = {'credible': 92, 'mostly_credible': 76, 'mixed': 50, 'unverifiable': 48, 'misleading': 22, 'false': 6}


def _fn_norm(v):
    return str(v or '').lower().strip().replace(' ', '_').replace('-', '_')


def _fn_recommend(score, satire, spoof, has_fact):
    if satire:
        return "😄 This looks like satire or parody. It is not meant to be real news."
    if spoof:
        return "🚨 A link imitates a trusted news site. Do not trust or share it until you check the real site yourself."
    if not has_fact:
        return ("⚠️ The web fact-check did not run, so this result only uses writing style, links and an AI text model. "
                "Search the main claim on Reuters, AP or BBC before trusting it.")
    if score >= 80:
        return "✓ The main claims check out against reliable sources. Still read the sources below before sharing."
    if score >= 60:
        return "👍 Mostly fine, but some parts are weak or unconfirmed. Check the flagged claims first."
    if score >= 40:
        return "⚠️ Mixed or unverified. Do not share it until the doubtful claims are confirmed by a reliable source."
    return "🚨 Serious problems found. Treat this as false or misleading and do not share it."


# ---------------- the new detect() ----------------
def _fn_detect(self, text):
    text = (text or '').strip()
    notes, fetched = [], None
    m = re.fullmatch(r"<?(https?://[^\s<>]+)>?", text)
    if m:
        fetched = m.group(1)
        title, body = _fn_fetch(fetched)           # raises a readable error if it fails
        text = (title + "\n\n" + body).strip()
        notes.append("I opened the link and read the article from " + _fn_host(fetched) + ".")
    own = _fn_host(fetched) if fetched else ''

    hosts = []
    for h in ([own] if own else []) + [_fn_host(u) for u in re.findall(_FN_URL, text)]:
        if h and h not in hosts:
            hosts.append(h)
    src = _fn_analyze_domains(hosts)

    style_pats, style = _fn_style(text)
    patterns = list(style_pats) + src['flags']
    ml = _fn_ml(self, text) if len(text) >= 80 else None

    # ---- web fact-check ----
    fact, web, searched = None, [], False
    bot = getattr(self, 'chatbot', None)
    if bot is None:
        notes.append("The chatbot is not connected, so no web fact-check was done.")
    else:
        try:
            fact, web, searched = bot.fact_check(text[:6000], fetched)
            if not searched:
                notes.append("Google Search was not available, so claims were checked from the AI's memory only. Trust this less.")
        except Exception as e:
            fact = None
            notes.append("The web fact-check could not run (" + str(e)[:120] + "). Check your Gemini key.")

    claims, gem, satire, overall = [], None, bool(src['satire']), ''
    if fact:
        overall = _fn_norm(fact.get('overall'))
        satire = satire or overall == 'satire' or 'satire' in str(fact.get('article_type', '')).lower()
        for c in (fact.get('claims') or [])[:8]:
            if isinstance(c, dict):
                v = _fn_norm(c.get('verdict'))
                claims.append({'claim': str(c.get('claim', ''))[:300],
                               'verdict': v if v in CLAIM_PTS or v == 'unverifiable' else 'unverifiable',
                               'evidence': str(c.get('evidence', ''))[:500]})
        base = OVERALL_PTS.get(overall)
        known = [CLAIM_PTS[c['verdict']] for c in claims if c['verdict'] in CLAIM_PTS]
        if known or base is not None:
            if known and base is not None:
                gem = 0.5 * base + 0.5 * (sum(known) / len(known))
            elif known:
                gem = sum(known) / len(known)
            else:
                gem = base
            try:
                conf = max(0, min(100, int(float(fact.get('confidence', 60)))))
            except Exception:
                conf = 60
            gem = 50 + (gem - 50) * (0.4 + 0.6 * conf / 100)
            if overall == 'false':
                gem = min(gem, 15)
            gem = int(round(gem))

    # ---- combine ----
    if gem is not None:
        w = {'gem': .60, 'style': .25, 'ml': .15}
    else:
        w = {'style': .60, 'ml': .40}
    vals = {'gem': gem, 'style': style, 'ml': ml}
    used = {k: v for k, v in w.items() if vals.get(k) is not None}
    score = sum(vals[k] * v for k, v in used.items()) / sum(used.values())

    own_trusted = bool(own) and _fn_is_trusted(own)
    if own_trusted:
        score += 8
    elif src['trusted_count']:
        score += 3
    if src['shortened']:
        score -= 5
    if src['sketchy_tld']:
        score -= 5
    if src['spoof_hosts']:
        score = min(score - 15, 25)
    has_fact = gem is not None
    if not has_fact:
        score = min(score, 68)
    if gem is not None and gem <= 15:
        score = min(score, 25)
    score = int(round(max(0, min(100, score))))

    if satire:
        verdict = "SATIRE / NOT REAL NEWS"
    elif score >= 80:
        verdict = "LIKELY CREDIBLE"
    elif score >= 60:
        verdict = "POSSIBLY CREDIBLE"
    elif score >= 40:
        verdict = "UNVERIFIED"
    elif score >= 20:
        verdict = "LIKELY FAKE"
    else:
        verdict = "FAKE / FALSE"

    if has_fact and searched:
        level = "High" if (fact and str(fact.get('confidence', 0)).replace('.', '').isdigit() and float(fact.get('confidence', 0)) >= 70) else "Medium"
    elif has_fact:
        level = "Low"
    else:
        level = "Low"
    if gem is not None and abs(gem - style) > 40:
        notes.append("The fact-check and the writing style disagree. Trust the fact-check more, because style alone can be wrong.")
    if ml is not None:
        notes.append("The AI text model was trained on older news, so it only counts for a small part of the score.")

    return {
        'success': True,
        'sources_found': hosts,
        'source_analysis': src,
        'suspicious_patterns': patterns,
        'sentiment_analysis': None,
        'credibility_score': score,
        'verdict': verdict,
        'confidence_level': level,
        'is_satire': satire,
        'recommendation': _fn_recommend(score, satire, bool(src['spoof_hosts']), has_fact),
        'fact_check': {
            'summary': str(fact.get('summary', '')) if fact else '',
            'article_type': str(fact.get('article_type', '')) if fact else '',
            'overall': overall,
            'claims': claims,
            'manipulation': [str(x) for x in (fact.get('manipulation') or [])][:8] if fact else [],
            'missing_context': str(fact.get('missing_context', '')) if fact else '',
        } if fact else None,
        'web_sources': web[:8],
        'signals': {'fact_check': gem, 'style': style, 'ai_model': ml},
        'fetched_from': fetched,
        'notes': notes,
    }


FakeNewsDetector.detect = _fn_detect

_fn_prev_init = WebUI.__init__


def _fn_init(self):
    _fn_prev_init(self)
    self.fake_news_detector.chatbot = self.chatbot


WebUI.__init__ = _fn_init

# ---------------- result screen ----------------
FN_CSS = r"""
#news-output{white-space:normal}
#news-output table{width:100%;border-collapse:collapse;margin:8px 0}
#news-output td,#news-output th{padding:6px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.2)}
#news-output ul{margin:0;padding-left:22px}
#news-output li{margin-bottom:6px}
.cr-status.bad{background:rgba(255,70,70,.15);border:1px solid rgba(255,70,70,.6);color:#ff9a9a}
body.light .cr-status.bad{color:#9b1c1c;background:rgba(255,70,70,.22)}
.fn-score{font-size:2.2em;font-weight:800;line-height:1.1}
.fn-score small{font-size:.4em;font-weight:400;opacity:.8;margin-left:8px}
.fn-meter{height:16px;border-radius:10px;background:rgba(255,255,255,.18);overflow:hidden;margin:8px 0 12px}
.fn-meter i{display:block;height:100%;border-radius:10px;transition:width .8s}
.fn-claim{padding:10px 14px;margin:8px 0;border-radius:12px;background:rgba(0,0,0,.25);border-left:5px solid #888}
.fn-claim.ok{border-left-color:#22c55e}.fn-claim.warn{border-left-color:#f59e0b}.fn-claim.bad{border-left-color:#ef4444}
.fn-badge{font-weight:700;margin-right:6px}
.fn-claim small{display:block;opacity:.85;margin-top:4px}
.fn-links a{color:inherit;text-decoration:underline;word-break:break-all}
body.light .fn-claim{background:rgba(255,255,255,.55)}
"""

FN_JS = r"""
(function () {
  const esc = s => escapeHtml(String(s));
  const title = t => '<div class="cr-title">' + t + '</div>';
  const list = a => '<ul>' + a.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul>';
  const VC = { true: ['✅', 'True', 'ok'], mostly_true: ['🟢', 'Mostly true', 'ok'], misleading: ['🟠', 'Misleading', 'warn'],
               false: ['❌', 'False', 'bad'], unverifiable: ['❔', 'Could not verify', 'warn'] };

  function render(d) {
    const sc = d.credibility_score;
    const col = sc >= 80 ? '#22c55e' : sc >= 60 ? '#84cc16' : sc >= 40 ? '#f59e0b' : '#ef4444';
    const cls = d.is_satire ? 'warn' : sc >= 60 ? 'ok' : sc >= 40 ? 'warn' : 'bad';
    let h = '<div class="cr-status ' + cls + '">' + esc(d.verdict) +
            ' <span style="font-weight:400">· confidence in this result: ' + esc(d.confidence_level) + '</span></div>';
    h += '<div class="fn-score">' + sc + '<small>/ 100 credibility</small></div>' +
         '<div class="fn-meter"><i style="width:' + sc + '%;background:' + col + '"></i></div>';
    if (d.fetched_from) h += '<p><small>Article read from: ' + esc(d.fetched_from) + '</small></p>';

    const f = d.fact_check;
    if (f && f.summary) h += title('🔎 Summary') + '<p>' + esc(f.summary) + '</p>';
    if (f && f.claims && f.claims.length) {
      h += title('🧾 Claims checked against the web') + f.claims.map(c => {
        const v = VC[c.verdict] || VC.unverifiable;
        return '<div class="fn-claim ' + v[2] + '"><span class="fn-badge">' + v[0] + ' ' + v[1] + '</span>' + esc(c.claim) +
               (c.evidence ? '<small>' + esc(c.evidence) + '</small>' : '') + '</div>';
      }).join('');
    }
    if (f && f.missing_context) h += title('🧩 Missing context') + '<p>' + esc(f.missing_context) + '</p>';

    const flags = (d.suspicious_patterns || []).concat((f && f.manipulation) || []);
    h += title('🚩 Red flags') + (flags.length ? list(flags) : '<p>✓ No suspicious patterns found.</p>');

    const sa = d.source_analysis || {};
    let srcBits = [];
    if (d.sources_found && d.sources_found.length) srcBits = d.sources_found.map(s => s);
    h += title('🔗 Links and sources');
    if (srcBits.length) {
      h += list(srcBits);
      if (sa.trusted_count) h += '<p>✓ ' + sa.trusted_count + ' trusted domain(s).</p>';
      if (sa.satire && sa.satire.length) h += '<p>😄 Known satire site: ' + esc(sa.satire.join(', ')) + '</p>';
    } else {
      h += '<p>No links in the text.</p>';
    }

    const ws = (d.web_sources || []).filter(s => /^https?:\/\//.test(s.url));
    if (ws.length) {
      h += title('🌐 Web evidence used') + '<ul class="fn-links">' + ws.map(s =>
        '<li><a href="' + esc(s.url) + '" target="_blank" rel="noopener noreferrer">' + esc(s.title) + '</a></li>').join('') + '</ul>';
    }

    const sg = d.signals || {};
    const cell = v => (v === null || v === undefined) ? '—' : v + '/100';
    h += title('📊 How the score was made') + '<table>' +
      '<tr><td>Web fact-check (main part)</td><td>' + cell(sg.fact_check) + '</td></tr>' +
      '<tr><td>Writing style</td><td>' + cell(sg.style) + '</td></tr>' +
      '<tr><td>AI text model</td><td>' + cell(sg.ai_model) + '</td></tr></table>';

    h += '<div class="img-tip">💡 ' + esc(d.recommendation) + '</div>';
    if (d.notes && d.notes.length) h += title('📝 Notes') + list(d.notes);
    h += '<div class="ppl-note">No tool can be 100% right. Breaking news may not be covered by sources yet, so check the evidence links yourself.</div>';
    return h;
  }

  window.detectFakeNews = async function () {
    const text = document.getElementById('news-input').value;
    const out = document.getElementById('news-output');
    if (!text || !text.trim()) { out.textContent = 'Please paste an article or a link.'; return; }
    out.innerHTML = '<div class="an-stage"><span>Fact-checking</span><span class="an-dots"><i></i><i></i><i></i></span></div>' +
                    '<div class="an-hint">Searching the web for each claim. This can take 10 to 40 seconds.</div>';
    try {
      const res = await fetch('/api/detect_fake_news', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text })
      });
      const data = await res.json();
      out.innerHTML = data.success ? render(data) : esc(data.error || 'Something went wrong.');
      out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
      out.textContent = 'Error: ' + e.message;
    }
  };
})();
"""

_fn_prev_get_html = WebUI.get_html


def _fn_get_html(self) -> str:
    html = _fn_prev_get_html(self)
    html = html.replace('</head>', '<style>' + FN_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + FN_JS + '</script></body>', 1)
    html = html.replace('Paste the news article text here...', 'Paste the article text, or just paste a link...', 1)
    return html


WebUI.get_html = _fn_get_html

# ============================================================================
# 3D STORM PATCH (dark mode): perspective rain, ground splashes, volumetric
# clouds, branching lightning with flicker + synthesized thunder.
# ============================================================================

STORM_CSS = r"""
#stormy-canvas{display:none!important}
#storm3d{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;display:block}
body.light #storm3d{display:none}
#stars-wrap{opacity:.3}
#stormSound{position:fixed;top:104px;right:38px;z-index:60}
body.light #stormSound{display:none}
"""

STORM_JS = r"""
(function () {
  const cv = document.getElementById('storm3d');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  if (!ctx) return;

  const isLight = () => document.body.classList.contains('light');
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

  // ---------- world / camera ----------
  const CH = 1.6;            // camera height (metres)
  const ZMIN = 1.1, ZMAX = 16, NB = 10;
  const TR = 0.04;           // streak length in seconds of motion
  let W = 0, H = 0, dpr = 1, HOR = 0, F = 1;
  let drops = [], splashes = [], sparks = [], bolts = [];
  let wind = 2, gust = 0, camX = 0, camTarget = 0, flash = 0;
  const bz = [];
  for (let b = 0; b < NB; b++) bz.push(ZMIN + (b + 0.5) * (ZMAX - ZMIN) / NB);

  // ---------- cached gradients / ridges ----------
  let skyGrad, groundGrad, sheenGrad, fogGrad, vig, ridgeCv;

  function makeRidges() {
    ridgeCv = document.createElement('canvas');
    ridgeCv.width = cv.width; ridgeCv.height = cv.height;
    const g = ridgeCv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const layers = [
      { amp: .07, h: .10, col: '#151c29', f: [.0035, .009, .021], j: 0 },
      { amp: .05, h: .06, col: '#0e131c', f: [.005, .013, .03], j: 2 },
      { amp: .03, h: .03, col: '#070a10', f: [.008, .02, .05], j: 4 }
    ];
    layers.forEach(L => {
      const p = [rand(0, 6.28), rand(0, 6.28), rand(0, 6.28)];
      g.fillStyle = L.col;
      g.beginPath(); g.moveTo(0, HOR + 6);
      for (let x = 0; x <= W; x += 3) {
        const n = Math.sin(x * L.f[0] + p[0]) * .55 + Math.sin(x * L.f[1] + p[1]) * .3 + Math.sin(x * L.f[2] + p[2]) * .15;
        g.lineTo(x, HOR - L.h * H * .5 - n * L.amp * H * .5 - Math.random() * L.j);
      }
      g.lineTo(W, HOR + 6); g.closePath(); g.fill();
    });
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    W = window.innerWidth; H = window.innerHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    HOR = H * 0.64;
    F = (H - HOR) * ZMIN / CH * 1.05;

    skyGrad = ctx.createLinearGradient(0, 0, 0, HOR);
    skyGrad.addColorStop(0, '#04060b'); skyGrad.addColorStop(.55, '#111826'); skyGrad.addColorStop(1, '#2a3547');
    groundGrad = ctx.createLinearGradient(0, HOR, 0, H);
    groundGrad.addColorStop(0, '#1b2332'); groundGrad.addColorStop(1, '#05070b');
    sheenGrad = ctx.createLinearGradient(0, HOR, 0, HOR + (H - HOR) * .55);
    sheenGrad.addColorStop(0, 'rgba(150,175,225,.55)'); sheenGrad.addColorStop(1, 'rgba(150,175,225,0)');
    fogGrad = ctx.createLinearGradient(0, HOR - 130, 0, HOR + 70);
    fogGrad.addColorStop(0, 'rgba(70,85,110,0)'); fogGrad.addColorStop(.62, 'rgba(78,93,120,.38)'); fogGrad.addColorStop(1, 'rgba(40,50,66,0)');
    vig = ctx.createRadialGradient(W / 2, H / 2, H * .42, W / 2, H / 2, Math.hypot(W, H) / 2);
    vig.addColorStop(0, 'rgba(0,0,0,0)'); vig.addColorStop(1, 'rgba(0,0,0,.62)');
    makeRidges();

    const want = clamp(Math.round(W * H / 620), 500, 1800);
    while (drops.length < want) { const d = {}; spawn(d, true); drops.push(d); }
    drops.length = Math.min(drops.length, want);
  }

  function spawn(d, initial) {
    d.z = ZMIN + (ZMAX - ZMIN) * Math.pow(Math.random(), 0.75);
    d.b = Math.min(NB - 1, Math.floor((d.z - ZMIN) / (ZMAX - ZMIN) * NB));
    d.vy = rand(8.5, 11.5);
    d.top = CH + (HOR + 60) * d.z / F;
    d.hy = initial ? rand(0, d.top) : d.top + rand(0, 1.5);
    d.xw = (Math.random() * 2 - 1) * (W * 0.5 + 80) * d.z / F;
    d.sx = 0; d.sy = -1000;
  }

  // ---------- tileable cloud textures ----------
  function makeCloudTex(w, h) {
    const octs = [[3, 2, 1], [6, 4, .55], [12, 8, .3], [24, 16, .17], [48, 32, .09], [96, 64, .05]];
    const grids = octs.map(o => { const a = new Float32Array(o[0] * o[1]); for (let i = 0; i < a.length; i++) a[i] = Math.random(); return a; });
    let norm = 0; octs.forEach(o => norm += o[2]);
    const field = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let v = 0;
        for (let o = 0; o < octs.length; o++) {
          const gx = octs[o][0], gy = octs[o][1], gr = grids[o];
          const fx = x / w * gx, fy = y / h * gy;
          const x0 = Math.floor(fx), y0 = Math.floor(fy);
          const tx = fx - x0, ty = fy - y0;
          const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
          const x1 = (x0 + 1) % gx, y1 = (y0 + 1) % gy;
          const a = gr[y0 * gx + x0], b = gr[y0 * gx + x1], c = gr[y1 * gx + x0], d = gr[y1 * gx + x1];
          const top = a + (b - a) * sx, bot = c + (d - c) * sx;
          v += octs[o][2] * (top + (bot - top) * sy);
        }
        field[y * w + x] = v / norm;
      }
    }
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const g = c.getContext('2d');
    const img = g.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const v = field[i] + 0.10 * (1 - y / h);
        const d = smooth(.42, .64, v) * (1 - smooth(.6, 1, y / h));
        const sh = field[((y - 4 + h) % h) * w + ((x - 4 + w) % w)];
        const light = clamp(.5 + (field[i] - sh) * 9, 0, 1);
        const base = 28 + light * 62 - d * 8;
        const p = i * 4;
        img.data[p] = base * .86; img.data[p + 1] = base * .94; img.data[p + 2] = Math.min(255, base * 1.2 + 6);
        img.data[p + 3] = d * 235;
      }
    }
    g.putImageData(img, 0, 0);
    return c;
  }

  let texA, texB;
  function tile(tex, y, h, tw, off, alpha) {
    ctx.globalAlpha = alpha;
    let x = -(((off % tw) + tw) % tw);
    for (; x < W; x += tw) ctx.drawImage(tex, x, y, tw, h);
  }

  // ---------- lightning ----------
  function bolt(x0, y0, x1, y1, spread, gen, cls, segs) {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
    if (gen >= 7 || len < 7) { segs[cls].push(x0, y0, x1, y1); return; }
    const nx = -dy / len, ny = dx / len;
    const off = (Math.random() - .5) * len * spread;
    const px = (x0 + x1) / 2 + nx * off, py = (y0 + y1) / 2 + ny * off;
    bolt(x0, y0, px, py, spread, gen + 1, cls, segs);
    bolt(px, py, x1, y1, spread, gen + 1, cls, segs);
    if (cls < 2 && gen >= 2 && gen <= 4 && Math.random() < .2) {
      const ang = Math.atan2(dy, dx) + (Math.random() < .5 ? -1 : 1) * rand(.25, .8);
      const bl = len * rand(1.4, 3.2);
      bolt(px, py, px + Math.cos(ang) * bl, py + Math.sin(ang) * bl, spread, gen + 2, cls + 1, segs);
    }
  }

  function makePulses() {
    const list = [{ at: 0, amp: 1, dec: .07 }];
    let t = rand(.07, .12);
    const n = Math.floor(rand(1, 4));
    for (let i = 0; i < n; i++) { list.push({ at: t, amp: rand(.45, 1), dec: rand(.06, .14) }); t += rand(.06, .16); }
    return { list, end: t + .5 };
  }

  function strike() {
    const sheet = Math.random() < .33;
    const pu = makePulses();
    const b = { t: 0, sheet, pulses: pu.list, end: pu.end, cur: 0, segs: [[], [], []], bright: 1, z: 8, w: 1, gx: 0, gy: 0, x1: 0, y1: 0, bk: NB - 1 };
    if (sheet) {
      b.z = rand(5, 14); b.gx = rand(.1, .9) * W; b.gy = rand(.04, .38) * HOR; b.bright = rand(.5, .9);
    } else {
      b.z = rand(2.6, 13);
      const k = F / b.z;
      const sx = W / 2 + (((Math.random() * 2 - 1) * (W * .42) * b.z / F)) * k;
      const x0 = sx + rand(-.08, .08) * W, y0 = rand(.03, .25) * HOR;
      const x1 = sx, y1 = HOR + CH * k;
      b.gx = x0; b.gy = y0; b.x1 = x1; b.y1 = y1;
      b.bright = clamp(1.25 - b.z * .06, .45, 1);
      b.w = clamp(4.2 / Math.pow(b.z, .75), .9, 2.2);
      bolt(x0, y0, x1, y1, .34, 0, 0, b.segs);
    }
    b.bk = Math.min(NB - 1, Math.floor((b.z - ZMIN) / (ZMAX - ZMIN) * NB));
    bolts.push(b);
    gust += rand(.5, 1.5);
    const d01 = clamp((b.z - 2.6) / 10.4, 0, 1);
    setTimeout(() => thunder(d01), ((b.z - 2) * .35 + .4) * 1000);
  }

  function drawBolt(b) {
    if (b.sheet) return;
    const inten = b.cur * b.bright;
    if (inten < .02) return;
    ctx.globalCompositeOperation = 'lighter';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    const WID = [1, .55, .3];
    const passes = [[9, .10, 'rgb(110,140,255)'], [3.4, .30, 'rgb(170,195,255)'], [1, 1, 'rgb(255,255,255)']];
    for (const [mul, a, col] of passes) {
      ctx.strokeStyle = col; ctx.globalAlpha = clamp(a * inten, 0, 1);
      for (let c = 0; c < 3; c++) {
        const s = b.segs[c]; if (!s.length) continue;
        ctx.lineWidth = Math.max(.5, b.w * WID[c] * mul);
        ctx.beginPath();
        for (let i = 0; i < s.length; i += 4) { ctx.moveTo(s[i], s[i + 1]); ctx.lineTo(s[i + 2], s[i + 3]); }
        ctx.stroke();
      }
    }
    // glow where it hits the ground
    const r = 150 / Math.pow(b.z, .6);
    ctx.save(); ctx.translate(b.x1, b.y1); ctx.scale(1, .28);
    const gg = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    gg.addColorStop(0, 'rgba(200,220,255,.9)'); gg.addColorStop(1, 'rgba(100,130,220,0)');
    ctx.fillStyle = gg; ctx.globalAlpha = clamp(inten * .5, 0, 1);
    ctx.fillRect(-r, -r, r * 2, r * 2);
    ctx.restore();
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
  }

  // ---------- sound (off until the button is clicked) ----------
  let audio = null, master = null, rainGain = null, noiseBuf = null, whiteBuf = null, soundOn = false;
  function initAudio() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;
    audio = new AC();
    master = audio.createGain(); master.gain.value = .9; master.connect(audio.destination);
    const sr = audio.sampleRate;
    noiseBuf = audio.createBuffer(1, sr * 9, sr);
    let d = noiseBuf.getChannelData(0), last = 0;
    for (let i = 0; i < d.length; i++) { last = (last + .02 * (Math.random() * 2 - 1)) / 1.02; d[i] = last * 3.2; }
    whiteBuf = audio.createBuffer(1, sr * 2, sr);
    d = whiteBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const rs = audio.createBufferSource(); rs.buffer = whiteBuf; rs.loop = true;
    const hp = audio.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 700;
    const lp = audio.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 7000;
    rainGain = audio.createGain(); rainGain.gain.value = 0;
    rs.connect(hp); hp.connect(lp); lp.connect(rainGain); rainGain.connect(master);
    rs.start();
    return true;
  }
  function applyRainVolume() {
    if (rainGain) rainGain.gain.setTargetAtTime((soundOn && !isLight()) ? .05 : 0, audio.currentTime, .3);
  }
  function thunder(d01) {
    if (!audio || !soundOn || isLight()) return;
    const t0 = audio.currentTime;
    const dur = 3.2 + (1 - d01) * 1.8 + rand(0, 1.2);
    const vol = .95 - .7 * d01;
    const src = audio.createBufferSource(); src.buffer = noiseBuf;
    const lp = audio.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = .7;
    lp.frequency.setValueAtTime(1400 - 900 * d01, t0);
    lp.frequency.exponentialRampToValueAtTime(70, t0 + dur);
    const g = audio.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + .03 + d01 * .3);
    let t = t0 + .3 + d01 * .3;
    while (t < t0 + dur - .6) { g.gain.setTargetAtTime(vol * rand(.25, .8), t, .25); t += rand(.35, .8); }
    g.gain.setTargetAtTime(0, t0 + dur - .6, .5);
    src.connect(lp); lp.connect(g); g.connect(master);
    src.start(t0, rand(0, 2)); src.stop(t0 + dur + 1.5);
    if (d01 < .4) {                                   // sharp crack for close strikes
      const cs = audio.createBufferSource(); cs.buffer = whiteBuf;
      const hp = audio.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 1500;
      const cg = audio.createGain();
      cg.gain.setValueAtTime(.55 * vol, t0); cg.gain.exponentialRampToValueAtTime(.001, t0 + .22);
      cs.connect(hp); hp.connect(cg); cg.connect(master);
      cs.start(t0); cs.stop(t0 + .3);
    }
  }

  const sbtn = document.createElement('button');
  sbtn.id = 'stormSound'; sbtn.type = 'button'; sbtn.className = 'mic-btn mini';
  sbtn.title = 'Rain and thunder sound'; sbtn.textContent = '🔇';
  document.body.appendChild(sbtn);
  sbtn.addEventListener('click', () => {
    if (!audio && !initAudio()) { alert('Sound is not supported in this browser.'); return; }
    audio.resume();
    soundOn = !soundOn;
    sbtn.textContent = soundOn ? '🔊' : '🔇';
    sbtn.classList.toggle('on', soundOn);
    applyRainVolume();
  });

  // ---------- input ----------
  const onMove = e => {
    const x = e.clientX !== undefined ? e.clientX : (e.touches && e.touches[0] ? e.touches[0].clientX : W / 2);
    camTarget = (x / W - .5) * 1.2;
  };
  window.addEventListener('mousemove', onMove);
  window.addEventListener('touchmove', onMove, { passive: true });
  window.addEventListener('resize', resize);

  // ---------- init ----------
  resize();
  texA = makeCloudTex(512, 256);
  texB = makeCloudTex(512, 256);

  // ---------- main loop ----------
  let last = performance.now(), T = 0, nextStrike = 1.8, lastLight = null;

  function frame(now) {
    requestAnimationFrame(frame);
    const light = isLight();
    if (light !== lastLight) { lastLight = light; applyRainVolume(); }
    if (light || document.hidden) { last = now; return; }
    const dt = Math.min(.05, Math.max(0, (now - last) / 1000)); last = now; T += dt;

    camX += (camTarget - camX) * Math.min(1, dt * 2.5);
    gust *= Math.exp(-dt * .8);
    wind = 2 + 1.1 * Math.sin(T * .35) + .7 * Math.sin(T * 1.3 + 1.7) + gust;

    // lightning schedule + flicker
    nextStrike -= dt;
    if (nextStrike <= 0) { strike(); nextStrike = Math.random() < .3 ? rand(.25, 1.1) : rand(3, 9); }
    flash = 0;
    for (let i = bolts.length - 1; i >= 0; i--) {
      const b = bolts[i]; b.t += dt;
      if (b.t > b.end) { bolts.splice(i, 1); continue; }
      let v = 0;
      for (const p of b.pulses) if (b.t >= p.at) v = Math.max(v, p.amp * Math.exp(-(b.t - p.at) / p.dec));
      b.cur = v * (.88 + Math.random() * .12);
      flash = Math.max(flash, b.cur * b.bright);
    }

    // ---- sky + clouds ----
    ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1;
    ctx.fillStyle = skyGrad; ctx.fillRect(0, 0, W, HOR + 2);
    tile(texA, 0, HOR * .78, W * 1.3, T * 5 + camX * 18, .75);
    tile(texA, -HOR * .04, HOR * .95, W * 2.0, T * 11 + camX * 40, .8);
    tile(texB, -HOR * .18, HOR * .85, W * 3.0, T * 24 + camX * 80, .9);

    // ---- flash lights the sky and the clouds ----
    if (flash > .01) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 1;
      ctx.fillStyle = 'rgba(110,130,190,' + (flash * .16).toFixed(3) + ')';
      ctx.fillRect(0, 0, W, HOR + 2);
      for (const b of bolts) {
        const i = b.cur * b.bright; if (i < .02) continue;
        const rad = W * (b.sheet ? .55 : .7);
        const gr = ctx.createRadialGradient(b.gx, b.gy, 0, b.gx, b.gy, rad);
        gr.addColorStop(0, 'rgba(210,225,255,' + (i * .55).toFixed(3) + ')');
        gr.addColorStop(.35, 'rgba(130,155,230,' + (i * .2).toFixed(3) + ')');
        gr.addColorStop(1, 'rgba(60,80,150,0)');
        ctx.fillStyle = gr; ctx.fillRect(0, 0, W, HOR + 2);
      }
      ctx.globalCompositeOperation = 'source-over';
    }

    // ---- hills, ground, mist ----
    ctx.globalAlpha = 1;
    ctx.drawImage(ridgeCv, 0, 0, W, H);
    ctx.fillStyle = groundGrad; ctx.fillRect(0, HOR, W, H - HOR);
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = clamp(.06 + flash * .55, 0, .8);
    ctx.fillStyle = sheenGrad; ctx.fillRect(0, HOR, W, (H - HOR) * .55);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = fogGrad; ctx.fillRect(0, HOR - 130, W, 200);
    tile(texA, HOR - 110, 170, W * 2.4, T * 16 + camX * 60, .2);
    ctx.globalAlpha = 1;

    // ---- update rain ----
    for (const d of drops) {
      d.hy -= d.vy * dt;
      d.xw += wind * dt;
      const k = F / d.z;
      if (d.hy <= 0) {
        if (d.z < 9 && splashes.length < 260 && Math.random() < .6) {
          splashes.push({ xw: d.xw, z: d.z, age: 0, life: rand(.28, .5), rw: rand(.08, .2) });
          if (d.z < 5 && sparks.length < 300) {
            for (let i = 0; i < 3; i++) sparks.push({ xw: d.xw, z: d.z, hy: 0, vx: rand(-1.2, 1.2), vy: rand(1.5, 3.2) });
          }
        }
        spawn(d, false);
        continue;
      }
      d.sx = W / 2 + (d.xw - camX) * k;
      d.sy = HOR + (CH - d.hy) * k;
      const lim = (W / 2 + 80) * d.z / F;
      if (d.xw - camX > lim) d.xw -= 2 * lim; else if (d.xw - camX < -lim) d.xw += 2 * lim;
    }

    // ---- splashes + droplets ----
    ctx.strokeStyle = 'rgb(190,208,235)'; ctx.lineWidth = 1;
    for (let i = splashes.length - 1; i >= 0; i--) {
      const s = splashes[i]; s.age += dt;
      if (s.age >= s.life) { splashes[i] = splashes[splashes.length - 1]; splashes.pop(); continue; }
      const t = s.age / s.life, k = F / s.z;
      const x = W / 2 + (s.xw - camX) * k, y = HOR + CH * k;
      const rx = s.rw * k * (.3 + .7 * t);
      ctx.globalAlpha = clamp((1 - t) * .55 * Math.exp(-.2 * (s.z - 1)) * (1 + flash * 1.6), 0, .85);
      ctx.beginPath(); ctx.ellipse(x, y, rx, Math.max(.4, rx * .3), 0, 0, 6.283); ctx.stroke();
    }
    ctx.fillStyle = 'rgb(200,215,240)';
    for (let i = sparks.length - 1; i >= 0; i--) {
      const p = sparks[i];
      p.vy -= 9.8 * dt; p.hy += p.vy * dt; p.xw += p.vx * dt;
      if (p.hy < 0) { sparks[i] = sparks[sparks.length - 1]; sparks.pop(); continue; }
      const k = F / p.z;
      ctx.globalAlpha = clamp(.55 * Math.exp(-.2 * (p.z - 1)) * (1 + flash), 0, .8);
      ctx.fillRect(W / 2 + (p.xw - camX) * k, HOR + (CH - p.hy) * k, 1.6, 1.6);
    }

    // ---- rain, far to near, with bolts drawn at their own depth ----
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgb(180,200,230)';
    for (let b = NB - 1; b >= 0; b--) {
      const zm = bz[b], k = F / zm;
      const a = clamp(.5 * Math.exp(-.16 * (zm - 1)) * (1 + flash * 2.4), 0, .9);
      const lw = .5 + 1.5 / zm;
      ctx.beginPath();
      for (const d of drops) {
        if (d.b !== b || d.sy < -60) continue;
        const kk = F / d.z;
        ctx.moveTo(d.sx, d.sy);
        ctx.lineTo(d.sx - wind * kk * TR, d.sy - d.vy * kk * TR);
      }
      if (zm < 4) { ctx.globalAlpha = a * .28; ctx.lineWidth = lw * 3.2; ctx.stroke(); }
      ctx.globalAlpha = a; ctx.lineWidth = lw; ctx.stroke();
      for (const bt of bolts) if (bt.bk === b) drawBolt(bt);
    }

    // ---- vignette ----
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = vig; ctx.fillRect(0, 0, W, H);
  }
  requestAnimationFrame(frame);
})();
"""

_storm_prev_get_html = WebUI.get_html


def _storm_get_html(self) -> str:
    html = _storm_prev_get_html(self)
    # stop the old flat storm animation (its canvas is hidden by the CSS above)
    html = re.sub(r"drawBackground\(\);\s*updateStorm\(\);", "", html, count=1)
    html = html.replace('<canvas id="stormy-canvas"></canvas>',
                        '<canvas id="stormy-canvas"></canvas><canvas id="storm3d"></canvas>', 1)
    html = html.replace('</head>', '<style>' + STORM_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + STORM_JS + '</script></body>', 1)
    return html


WebUI.get_html = _storm_get_html

# ============================================================================
# GALAXY STARS PATCH (dark mode): milky-way band, colored nebula, dust lanes,
# realistic star colors + glow + twinkle, parallax drift, shooting stars.
# ============================================================================

GALAXY_CSS = r"""
.panel-stars .ps{display:none!important}
.pg-canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.tab-inner{background:radial-gradient(ellipse at 50% 120%,#171d3a 0%,#080a16 55%,#030409 100%)}
"""

GALAXY_JS = r"""
(function () {
  const layers = Array.from(document.querySelectorAll('.panel-stars'));
  if (!layers.length) return;
  const rand = (a, b) => a + Math.random() * (b - a);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;
  const DPR = Math.min(window.devicePixelRatio || 1, 1.5);

  // real star colors: blue-white -> white -> yellow -> orange
  const PAL = [[155,176,255],[175,195,255],[205,218,255],[245,247,255],[255,246,236],[255,234,196],[255,214,165],[255,186,135]];
  const NEB = [[110,70,220],[50,100,230],[210,70,180],[40,170,200],[140,90,255],[255,150,100]];

  const sprites = PAL.map(c => {
    const s = document.createElement('canvas'); s.width = s.height = 64;
    const g = s.getContext('2d');
    const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)');
    gr.addColorStop(.08, 'rgba(' + c + ',.95)');
    gr.addColorStop(.25, 'rgba(' + c + ',.35)');
    gr.addColorStop(1, 'rgba(' + c + ',0)');
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
    return s;
  });
  const tiny = PAL.map(c => 'rgb(' + c + ')');

  function blob(g, x, y, r, c, a) {
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, 'rgba(' + c + ',' + a + ')');
    gr.addColorStop(1, 'rgba(' + c + ',0)');
    g.fillStyle = gr; g.fillRect(x - r, y - r, r * 2, r * 2);
  }

  // one big pre-drawn galaxy picture (square, so it can rotate without gaps)
  function buildGalaxy(S) {
    const c = document.createElement('canvas'); c.width = c.height = S;
    const g = c.getContext('2d');
    const ang = rand(-.7, -.35), ca = Math.cos(ang), sa = Math.sin(ang);
    const at = (t, o) => [S / 2 + ca * t - sa * o, S / 2 + sa * t + ca * o];

    g.globalCompositeOperation = 'lighter';
    for (let i = 0; i < 14; i++) {                       // soft glow of the galactic disc
      const [x, y] = at(rand(-.45, .45) * S, gauss() * S * .05);
      blob(g, x, y, rand(.14, .3) * S, [255, 226, 200], rand(.035, .07));
    }
    for (let i = 0; i < 46; i++) {                       // colored nebula clouds
      const [x, y] = at(rand(-.5, .5) * S, gauss() * S * .1);
      blob(g, x, y, rand(.05, .16) * S, NEB[Math.floor(Math.random() * NEB.length)], rand(.04, .12));
    }
    const [cx, cy] = at(rand(-.1, .1) * S, 0);           // bright core
    blob(g, cx, cy, S * .16, [255, 235, 205], .16);
    blob(g, cx, cy, S * .07, [255, 248, 235], .18);

    const n = Math.round(S * S / 700), px = Math.max(1, DPR * .9);
    for (let i = 0; i < n; i++) {                        // thousands of tiny milky-way stars
      const [x, y] = at(rand(-.5, .5) * S, gauss() * S * (Math.random() < .75 ? .05 : .14));
      g.fillStyle = 'rgba(' + PAL[Math.floor(Math.random() * PAL.length)] + ',' + rand(.15, .75) + ')';
      g.fillRect(x, y, px, px);
    }

    g.globalCompositeOperation = 'source-over';          // dark dust lanes dim the star cloud
    for (let i = 0; i < 30; i++) {
      const [x, y] = at(rand(-.45, .45) * S, gauss() * S * .02);
      blob(g, x, y, rand(.02, .07) * S, [2, 3, 10], rand(.18, .38));
    }
    return c;
  }

  function makeStar(w, h) {
    const z = Math.random(), mag = Math.pow(Math.random(), 5);
    const sp = 2 + z * 9;                                // near stars drift faster (parallax)
    return { x: rand(0, w), y: rand(0, h), vx: -sp, vy: sp * .22,
             r: .35 + mag * 2.4, a: .45 + .55 * Math.max(mag, Math.random() * .6),
             ci: Math.floor(Math.pow(Math.random(), .8) * PAL.length),
             ts: rand(.6, 2.6), ph: rand(0, 6.28), td: rand(.15, .6) };
  }

  const states = layers.map(layer => {
    const cv = document.createElement('canvas'); cv.className = 'pg-canvas'; layer.appendChild(cv);
    return { layer, cv, g: cv.getContext('2d'), w: 0, h: 0, S: 0, gal: null, stars: [],
             shoot: null, nextShoot: rand(3, 8), ang: rand(0, 6.28) };
  });

  function fit(st) {
    const w = st.layer.clientWidth, h = st.layer.clientHeight;
    if (!w || !h) return false;
    if (Math.abs(w - st.w) < 1 && Math.abs(h - st.h) < 1) return true;
    st.w = w; st.h = h;
    st.cv.width = Math.round(w * DPR); st.cv.height = Math.round(h * DPR);
    st.g.setTransform(DPR, 0, 0, DPR, 0, 0);
    const need = Math.min(2600, Math.ceil(Math.hypot(w, h) * DPR * 1.05));
    if (!st.gal || need > st.S || need < st.S * .6) { st.S = need; st.gal = buildGalaxy(need); }
    const want = Math.round(clamp(w * h / 2600, 120, 520));
    while (st.stars.length < want) st.stars.push(makeStar(w, h));
    st.stars.length = Math.min(st.stars.length, want);
    return true;
  }

  let last = performance.now(), T = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(.05, Math.max(0, (now - last) / 1000)); last = now; T += dt;
    if (document.hidden || document.body.classList.contains('light')) return;

    for (const st of states) {
      if (!st.layer.offsetParent || !fit(st)) continue;   // skip hidden tabs
      const g = st.g, w = st.w, h = st.h;
      g.setTransform(DPR, 0, 0, DPR, 0, 0);
      g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1;
      g.clearRect(0, 0, w, h);

      // the galaxy slowly turns and breathes
      st.ang += dt * .006;
      const Sc = st.S / DPR;
      g.save();
      g.translate(w / 2 + Math.sin(T * .05) * 8, h / 2 + Math.cos(T * .04) * 6);
      g.rotate(st.ang);
      g.globalAlpha = .9 + .1 * Math.sin(T * .3);
      g.drawImage(st.gal, -Sc / 2, -Sc / 2, Sc, Sc);
      g.restore();

      // stars
      g.globalCompositeOperation = 'lighter';
      for (const s of st.stars) {
        s.x += s.vx * dt; s.y += s.vy * dt;
        if (s.x < -8) { s.x = w + 8; s.y = rand(0, h); }
        if (s.y > h + 8) { s.y = -8; s.x = rand(0, w); }
        if (s.x > w + 8) s.x = rand(0, w);
        const al = s.a * (1 - s.td + s.td * (.5 + .5 * Math.sin(T * s.ts + s.ph)));
        g.globalAlpha = al;
        if (s.r < .75) {
          g.fillStyle = tiny[s.ci];
          g.fillRect(s.x - .5, s.y - .5, 1.1, 1.1);
        } else {
          const sz = s.r * 7;
          g.drawImage(sprites[s.ci], s.x - sz / 2, s.y - sz / 2, sz, sz);
          if (s.r > 1.9) {                                // diffraction spikes on the brightest
            const L = s.r * 6;
            g.globalAlpha = al * .45; g.strokeStyle = '#fff'; g.lineWidth = .6;
            g.beginPath();
            g.moveTo(s.x - L, s.y); g.lineTo(s.x + L, s.y);
            g.moveTo(s.x, s.y - L); g.lineTo(s.x, s.y + L);
            g.stroke();
          }
        }
      }

      // shooting star now and then
      st.nextShoot -= dt;
      if (st.nextShoot <= 0 && !st.shoot) {
        const a = rand(.35, .7) * (Math.random() < .5 ? 1 : -1), sp = rand(420, 620);
        const dir = Math.random() < .5 ? 1 : -1;
        st.shoot = { x: rand(.15, .85) * w, y: rand(0, .35) * h, vx: Math.cos(a) * sp * dir, vy: Math.abs(Math.sin(a)) * sp, t: 0, life: rand(.7, 1.1) };
      }
      const sh = st.shoot;
      if (sh) {
        sh.t += dt;
        const k = sh.t / sh.life;
        if (k >= 1) { st.shoot = null; st.nextShoot = rand(5, 13); }
        else {
          const hx = sh.x + sh.vx * sh.t, hy = sh.y + sh.vy * sh.t, len = Math.hypot(sh.vx, sh.vy);
          const tx = hx - sh.vx / len * 90, ty = hy - sh.vy / len * 90;
          const gr = g.createLinearGradient(tx, ty, hx, hy);
          gr.addColorStop(0, 'rgba(180,200,255,0)'); gr.addColorStop(1, 'rgba(255,255,255,' + (1 - k) + ')');
          g.globalAlpha = 1; g.strokeStyle = gr; g.lineWidth = 1.6; g.lineCap = 'round';
          g.beginPath(); g.moveTo(tx, ty); g.lineTo(hx, hy); g.stroke();
          g.globalAlpha = 1 - k; g.drawImage(sprites[3], hx - 8, hy - 8, 16, 16);
        }
      }
      g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
    }
  }
  requestAnimationFrame(frame);
})();
"""

_galaxy_prev_get_html = WebUI.get_html


def _galaxy_get_html(self) -> str:
    html = _galaxy_prev_get_html(self)
    html = html.replace('</head>', '<style>' + GALAXY_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + GALAXY_JS + '</script></body>', 1)
    return html


WebUI.get_html = _galaxy_get_html

# ============================================================================
# VOICE v2 PATCH
#  * Whisper large-v3 (strongest) with fallbacks, fp16 + SDPA, preloaded + warmed up
#  * Accent fix: explicit language detection with probabilities, a language
#    dropdown filled automatically from the model, and a "same language as
#    last time" prior when unsure
#  * Filters: mic noise-suppression/echo-cancel (browser), high/low-pass +
#    silence trimming + level normalising (server), speech gate, hallucination /
#    repetition checks, retry with beam search, Gemini audio as rescue
#  * Chatbot knows the text came from speech and infers what you MEANT
#  * Speed: preload, auto-send when you stop talking, memory write in background
# ============================================================================
import subprocess
import zlib
import wave

_VOICE_TL = threading.local()

# Language names come from Whisper itself (about 99 languages), nothing is typed by hand
try:
    from transformers.models.whisper.tokenization_whisper import LANGUAGES as _WHISPER_LANGS
except Exception:
    _WHISPER_LANGS = {"en": "english", "hy": "armenian", "ru": "russian", "es": "spanish",
                      "fr": "french", "de": "german", "ar": "arabic", "zh": "chinese"}
LANG_NAMES = {code: name.title() for code, name in _WHISPER_LANGS.items()}

VOICE_HINT = (
    "\n\nVOICE MESSAGE: The user's message was transcribed from speech by an automatic recognizer. "
    "Speakers may have accents, so some words can be misheard (similar-sounding or unrelated words, "
    "wrong names, missing words). Work out what they most likely MEANT from the whole sentence and the "
    "conversation, and answer that. Never mention transcription errors. Only if the meaning is truly "
    "unclear, give your best guess and ask one short question. Keep the answer conversational and fairly "
    "short because it may be read aloud; give more detail only if asked."
)


def _wav_bytes(a, sr=16000):
    buf = io.BytesIO()
    w = wave.open(buf, 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((np.clip(a, -1, 1) * 32767).astype('<i2').tobytes())
    w.close()
    return buf.getvalue()


class VoiceTranscriber:
    """Whisper large-v3 speech-to-text tuned for accents, with filters and a Gemini audio rescue."""

    GPU_MODELS = ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "openai/whisper-small"]
    CPU_MODELS = ["openai/whisper-small", "openai/whisper-base"]
    SR = 16000
    CHUNK = 28 * 16000
    HALLU = ("thank you for watching", "thanks for watching", "please subscribe", "subtitles by",
             "subtitles made by", "amara.org", "like and subscribe", "see you in the next video",
             "transcribed by", "www.", "субтитры", "продолжение следует")

    def __init__(self):
        self.model = None
        self.proc = None
        self.model_name = None
        self.lang_ids = {}
        self.last_lang = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def _load(self):
        if self.model is not None:
            return True
        with self._load_lock:
            if self.model is not None:
                return True
            if not TRANSFORMERS_AVAILABLE:
                return False
            import gc
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            gpu = torch.cuda.is_available()
            names = ([os.getenv("VOICE_MODEL")] if os.getenv("VOICE_MODEL") else []) + \
                    (self.GPU_MODELS if gpu else self.CPU_MODELS)
            for name in names:
                try:
                    print(f"Loading speech model {name} ...")
                    proc = WhisperProcessor.from_pretrained(name)
                    kw = dict(torch_dtype=torch.float16 if gpu else torch.float32, low_cpu_mem_usage=True)
                    try:
                        model = WhisperForConditionalGeneration.from_pretrained(name, attn_implementation="sdpa", **kw)
                    except Exception:
                        model = WhisperForConditionalGeneration.from_pretrained(name, **kw)
                    model.to("cuda" if gpu else "cpu")
                    model.eval()
                    l2i = getattr(model.generation_config, "lang_to_id", None) or {}
                    self.lang_ids = {k[2:-2]: v for k, v in l2i.items()}
                    self.proc, self.model, self.model_name = proc, model, name
                    print(f"✓ Loaded {name} for voice messages")
                    return True
                except Exception as e:
                    print(f"Could not load {name}: {str(e)[:150]}")
                    gc.collect()
                    if gpu:
                        torch.cuda.empty_cache()
            return False

    def _warm(self):
        try:
            import torch
            z = np.random.randn(self.SR).astype(np.float32) * 0.01
            f = self._features([z])
            with torch.inference_mode():
                self.model.generate(input_features=f, language="en", task="transcribe", max_new_tokens=4)
            self._detect(f)
        except Exception as e:
            print("Voice warm-up skipped:", str(e)[:100])

    def preload(self):
        with self._lock:
            if self._load():
                self._warm()
                print("✓ Voice model ready, first voice message will be fast")

    # ------------------------------------------------------------------ audio
    def _decode(self, path):
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(self.SR),
               "-af", "highpass=f=70,lowpass=f=7800", "-f", "f32le", "pipe:1"]
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=60)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed. Run: !apt-get -qq install -y ffmpeg")
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError("Could not read the audio: " + p.stderr.decode(errors="ignore")[:120])
        return np.frombuffer(p.stdout, dtype=np.float32).copy()

    def _clean(self, a):
        """Light clean-up. Returns (audio, speech_seconds, stats). audio is None only for real silence."""
        dur = a.size / self.SR
        stats = {"seconds": round(dur, 2), "peak_db": -120.0, "rms_db": -120.0}
        if a.size < self.SR * 0.25:
            return None, 0.0, stats
        a = a - float(a.mean())
        peak = float(np.max(np.abs(a)))
        stats["peak_db"] = round(20 * np.log10(peak + 1e-9), 1)
        stats["rms_db"] = round(20 * np.log10(float(np.sqrt(np.mean(a ** 2) + 1e-12)) + 1e-9), 1)
        if peak < 0.003:                       # about -50 dB: the mic really sent silence
            return None, 0.0, stats

        # make quiet recordings louder (gain is capped)
        p = float(np.percentile(np.abs(a), 99.5))
        a = np.clip(a * min(30.0, max(0.5, 0.7 / max(p, 1e-6))), -1, 1).astype(np.float32)

        # trim silence at the start and end, but only if speech is clearly found
        fl = int(0.02 * self.SR)
        n = a.size // fl
        speech_s = dur
        if n >= 5:
            rms = np.sqrt((a[:n * fl].reshape(n, fl) ** 2).mean(1) + 1e-12)
            thr = max(float(np.percentile(rms, 10)) * 2.5, 0.02)
            voiced = np.convolve((rms > thr).astype(int), np.ones(3, int), 'same') >= 2
            idx = np.where(voiced)[0]
            if idx.size * 0.02 >= 0.2:
                s = max(0, idx[0] - 20)
                e = min(n, idx[-1] + 1 + 25)
                a = a[s * fl:e * fl]
                speech_s = float(voiced.sum() * 0.02)
        return a, speech_s, stats

    def _chunks(self, a):
        if a.size <= self.CHUNK:
            return [a]
        out, i, fl = [], 0, int(0.1 * self.SR)
        while a.size - i > self.CHUNK:
            lo, hi = i + int(self.CHUNK * 0.7), i + self.CHUNK
            seg = a[lo:hi]
            m = seg.size // fl
            en = np.abs(seg[:m * fl]).reshape(m, fl).mean(1)
            cut = lo + int(np.argmin(en)) * fl + fl // 2       # split in the quietest spot
            out.append(a[i:cut])
            i = cut
        out.append(a[i:])
        return out

    # ---------------------------------------------------------------- whisper
    def _features(self, chunks):
        f = self.proc.feature_extractor(chunks, sampling_rate=self.SR, return_tensors="pt").input_features
        return f.to(self.model.device, dtype=self.model.dtype)

    def _detect(self, feats):
        """Language probabilities (top 5), highest first."""
        import torch
        items = list(self.lang_ids.items())
        if not items:
            return {}
        start = self.model.generation_config.decoder_start_token_id
        dec = torch.tensor([[start]], device=self.model.device)
        with torch.inference_mode():
            logits = self.model(input_features=feats[:1], decoder_input_ids=dec).logits[0, -1].float()
        idx = torch.tensor([i for _, i in items], device=logits.device)
        p = torch.softmax(logits[idx], dim=-1).cpu().numpy()
        order = np.argsort(-p)[:5]
        return {items[j][0]: float(p[j]) for j in order}

    def _pick_lang(self, probs):
        if not probs:
            return None, 0.0
        top, p = next(iter(probs.items()))
        if p >= 0.75:
            return top, p
        # unsure (typical with accents): lean on the language used last time
        prior = self.last_lang
        if prior and prior in probs and probs[prior] >= 0.25 * p:
            return prior, probs[prior]
        return top, p

    def _run(self, feats, lang, beams):
        import torch
        kw = dict(task="transcribe", num_beams=beams, max_new_tokens=220)
        if lang:
            kw["language"] = lang
        with torch.inference_mode():
            ids = self.model.generate(input_features=feats, **kw)
        txt = self.proc.batch_decode(ids, skip_special_tokens=True)
        return " ".join(t.strip() for t in txt if t.strip()).strip()

    # ---------------------------------------------------------------- filters
    def _suspicious(self, text, speech_s):
        t = (text or "").strip()
        if len(re.sub(r"[\W_]+", "", t)) < 1:
            return "empty"
        low = t.lower()
        if len(t) < 100 and any(h in low for h in self.HALLU):
            return "hallucination"
        b = t.encode("utf-8")
        if len(b) > 60 and len(b) / len(zlib.compress(b)) > 2.4:
            return "repetition"
        if re.search(r"\b(\w+)(?:[\s,.]+\1\b){4,}", low):
            return "repetition"
        cps = len(t) / max(speech_s, 0.1)
        if speech_s >= 1.5 and cps < 1.5:
            return "too_little"
        if cps > 45:
            return "too_much"
        return None

    # ------------------------------------------------------------------ main
    def transcribe(self, path, lang_pref="auto", bot=None):
        audio = self._decode(path)
        audio, speech_s, st = self._clean(audio)
        print(f"Voice input: {st['seconds']}s, peak {st['peak_db']} dB, average {st['rms_db']} dB")
        if audio is None:
            raise ValueError(
                f"The microphone sent almost no sound (recorded {st['seconds']}s, peak {st['peak_db']} dB). "
                "Check that the right microphone is selected in the browser and that it is not muted, then try again.")

        # Language is LOCKED, so accents can never make it guess a wrong language.
        # Change it by setting the VOICE_LANGUAGE environment variable (default: English).
        lang = (os.getenv("VOICE_LANGUAGE", "en") or "en").strip().lower()

        text, source, reason = "", "whisper", None
        with self._lock:
            if self._load():
                feats = self._features(self._chunks(audio))
                text = self._run(feats, lang, 5)           # beam search = more careful with accents
                reason = self._suspicious(text, speech_s)

        if (reason or not text) and bot is not None:       # Gemini listens to the audio itself
            try:
                g = bot.transcribe_audio(_wav_bytes(audio), LANG_NAMES.get(lang, lang))
                if g and not self._suspicious(g, speech_s):
                    text, source, reason = g, "gemini", None
            except Exception as e:
                print("Gemini audio rescue failed:", str(e)[:150])

        if not text or reason in ("empty", "hallucination"):
            raise ValueError("I couldn't make out what you said. Try again a little slower or closer to the mic.")
        return {"text": text, "language": lang, "lang_confidence": 1.0, "source": source,
                "uncertain": bool(reason), "seconds": round(audio.size / self.SR, 1)}

# ---------------- chatbot: audio rescue, voice hint, faster memory ----------------
def _bot_transcribe_audio(self, wav_bytes, lang_name=None):
    if not self._ensure_client():
        raise RuntimeError(self.last_error or "Gemini is not available.")
    hint = f"The speaker most likely speaks {lang_name}. " if lang_name else ""
    prompt = ("Transcribe exactly what the speaker says in this audio. " + hint +
              "The speaker may have a strong accent, so use the context of the whole sentence to choose the right words. "
              "Keep the original language (do not translate). Return ONLY the transcript, nothing else. "
              "If there is no clear speech, return an empty string.")
    models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
    last, discovered = None, False
    while True:
        for m in models:
            try:
                resp = self.client.models.generate_content(
                    model=m,
                    contents=[genai_types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), prompt],
                    config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024))
                self.chat_model = m
                return (getattr(resp, "text", None) or "").strip()
            except Exception as e:
                last = f"{m}: {e}"
                self.last_error = last
        if discovered:
            break
        discovered = True
        found, _ = self._discover_models()
        models = [x for x in found if x not in models][:4]
        if not models:
            break
    raise RuntimeError(last or "Gemini could not transcribe the audio.")


SimpleChatbot.transcribe_audio = _bot_transcribe_audio

_vp_prev_bsp = SimpleChatbot._build_system_prompt


def _bsp_voice(self, message):
    prompt, qvec = _vp_prev_bsp(self, message)
    if getattr(_VOICE_TL, "on", False):
        prompt += VOICE_HINT
    return prompt, qvec


SimpleChatbot._build_system_prompt = _bsp_voice

_vp_prev_remember = SimpleChatbot._remember


def _remember_bg(self, user_text, bot_text):      # saves one network round trip per reply
    threading.Thread(target=_vp_prev_remember, args=(self, user_text, bot_text), daemon=True).start()


SimpleChatbot._remember = _remember_bg


# ---------------- routes ----------------
def _voice_route_v2(self):
    @self.app.route('/api/voice', methods=['POST'])
    def voice():
        f = request.files.get('audio')
        if not f:
            return jsonify({'success': False, 'error': 'No audio received.'}), 400
        lang_pref = (request.form.get('lang') or 'auto').strip().lower()
        mt = (f.mimetype or '').lower()
        suffix = '.ogg' if 'ogg' in mt else '.m4a' if ('mp4' in mt or 'aac' in mt) else '.wav' if 'wav' in mt else '.webm'
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp)
                path = tmp.name
            t0 = time.time()
            res = self.voice.transcribe(path, lang_pref, self.chatbot)
            stt = time.time() - t0
            _VOICE_TL.on = True
            try:
                reply = self.chatbot.chat(res["text"])
            finally:
                _VOICE_TL.on = False
            return jsonify({'success': True, 'transcript': res["text"], 'language': res["language"],
                            'language_name': LANG_NAMES.get(res["language"], res["language"] or ''),
                            'source': res["source"], 'uncertain': res["uncertain"],
                            'stt_seconds': round(stt, 1), 'response': reply})
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            print(f"Voice error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    @self.app.route('/api/voice_languages')
    def voice_languages():
        codes = list(self.voice.lang_ids) or list(LANG_NAMES)
        items = sorted(((c, LANG_NAMES.get(c, c.upper())) for c in codes), key=lambda x: x[1])
        return jsonify({'languages': [{'code': c, 'name': n} for c, n in items]})


WebUI._add_voice_route = _voice_route_v2

_voice2_prev_init = WebUI.__init__


def _voice2_init(self):
    _voice2_prev_init(self)
    if os.getenv("PRELOAD_VOICE", "1") == "1":
        threading.Thread(target=self.voice.preload, daemon=True).start()


WebUI.__init__ = _voice2_init

# ---------------- browser side: filters, language list, auto-send on silence ----------------
VOICE2_CSS = r"""
#voice-lang{width:auto;max-width:150px;padding:6px 10px;font-size:14px;border-radius:12px}
.voice-text small{opacity:.7;font-style:normal}
"""

VOICE2_JS = r"""
(function () {
  const oldMic = document.querySelector('#chat .composer .mic-btn');
  const speakBtn = document.querySelector('#chat .chat-head .mic-btn.mini');
  if (!oldMic || !speakBtn) return;
  const mic = oldMic.cloneNode(true);          // clone = drop the old click handler
  oldMic.replaceWith(mic);

  const SILENCE_MS = 1300, NO_SPEECH_MS = 9000, MAX_MS = 120000;

  const box = speakBtn.parentNode;
  const sel = document.createElement('select');
  sel.id = 'voice-lang';
  sel.title = 'The language you speak. Picking yours gives the best accuracy with accents.';
  sel.innerHTML = '<option value="auto">Auto-detect</option>';
  let savedLang = 'auto';
  try { savedLang = localStorage.getItem('pipVoiceLang') || 'auto'; } catch (e) {}
  sel.addEventListener('change', () => { try { localStorage.setItem('pipVoiceLang', sel.value); } catch (e) {} });

  // the full list of languages comes from the speech model itself
  fetch('/api/voice_languages').then(r => r.json()).then(d => {
    sel.innerHTML = '<option value="auto">Auto-detect</option>' +
      d.languages.map(l => '<option value="' + l.code + '">' + l.name + '</option>').join('');
    sel.value = savedLang;
    if (sel.value !== savedLang) sel.value = 'auto';
  }).catch(() => {});

  let autoOn = true;
  try { autoOn = localStorage.getItem('pipVoiceAuto') !== '0'; } catch (e) {}
  const autoBtn = document.createElement('button');
  autoBtn.type = 'button'; autoBtn.className = 'mic-btn mini'; autoBtn.textContent = '⚡';
  autoBtn.title = 'Auto-send when I stop talking';
  autoBtn.classList.toggle('on', autoOn);
  autoBtn.addEventListener('click', () => {
    autoOn = !autoOn; autoBtn.classList.toggle('on', autoOn);
    try { localStorage.setItem('pipVoiceAuto', autoOn ? '1' : '0'); } catch (e) {}
  });
  box.insertBefore(sel, speakBtn);
  box.insertBefore(autoBtn, speakBtn);

  function speak(text, lang) {
    if (!window.speechSynthesis) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/```[\s\S]*?```/g, ' code block. ').replace(/[*`#]/g, ''));
    u.lang = lang || navigator.language || 'en-US';
    speechSynthesis.speak(u);
  }

  let rec = null, stream = null, chunks = [], cancelled = false, actx = null, timer = null;
  const defaultPh = chatInput.placeholder;

  function cleanup() {
    clearInterval(timer); timer = null;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    if (actx) { try { actx.close(); } catch (e) {} actx = null; }
    mic.classList.remove('rec'); mic.textContent = '🎤';
    chatInput.placeholder = defaultPh;
  }
  function notice(msg) {
    setTimeout(() => { chatInput.placeholder = msg; setTimeout(() => { chatInput.placeholder = defaultPh; }, 2500); }, 300);
  }
  function stopRec(cancel) {
    cancelled = !!cancel;
    if (rec && rec.state === 'recording') rec.stop(); else cleanup();
  }

  async function startRec() {
    if (chatBusy) return;
    if (!navigator.mediaDevices || !window.MediaRecorder) { alert('Recording is not supported in this browser.'); return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
    } catch (e) { alert('Please allow microphone access to send voice messages.'); return; }

    chunks = []; cancelled = false;
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'].find(t => MediaRecorder.isTypeSupported(t)) || '';
    const opts = { audioBitsPerSecond: 64000 };
    if (mime) opts.mimeType = mime;
    rec = new MediaRecorder(stream, opts);
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
      const was = cancelled;
      cleanup();
      if (!was && blob.size > 800) sendVoice(blob);
    };
    rec.start();
    mic.classList.add('rec'); mic.textContent = '⏹';

    const t0 = Date.now();
    let heard = false, lastVoice = t0, loud = 0, thr = 0.02;
    const calib = [];
    let an = null, buf = null;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      actx = new AC();
      const src = actx.createMediaStreamSource(stream);
      an = actx.createAnalyser(); an.fftSize = 1024;
      src.connect(an);
      buf = new Float32Array(an.fftSize);
    } catch (e) { an = null; }

    timer = setInterval(() => {
      const now = Date.now(), el = now - t0;
      if (an) {
        an.getFloatTimeDomainData(buf);
        let s = 0;
        for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
        const rms = Math.sqrt(s / buf.length);
        if (el < 350) {
          calib.push(rms);
          thr = Math.min(0.05, Math.max(0.012, 3 * calib.reduce((a, b) => a + b, 0) / calib.length));
        } else if (rms > thr) {
          loud++;
          if (loud >= 3) heard = true;
          lastVoice = now;
        } else {
          loud = 0;
        }
        if (autoOn && heard && now - lastVoice > SILENCE_MS) { stopRec(false); return; }
        if (autoOn && !heard && el > NO_SPEECH_MS) { stopRec(true); notice('I did not hear anything. Tap 🎤 and try again.'); return; }
      }
      if (el > MAX_MS) { stopRec(false); return; }
      chatInput.placeholder = (heard ? 'Listening… ' : 'Say something… ') + Math.round(el / 1000) + 's ' +
        (autoOn ? '(sends when you pause, Esc cancels)' : '(tap ⏹ to send, Esc cancels)');
    }, 50);
  }

  mic.addEventListener('click', () => { (rec && rec.state === 'recording') ? stopRec(false) : startRec(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && rec && rec.state === 'recording') stopRec(true); });

  async function sendVoice(blob) {
    chatBusy = true; chatSendBtn.disabled = true; mic.disabled = true;
    const u = addRow('user');
    u.bubble.innerHTML = '<audio controls src="' + URL.createObjectURL(blob) + '"></audio><div class="voice-text">Transcribing…</div>';
    const bot = addRow('assistant');
    bot.bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
    scrollChat();
    const done = () => { chatBusy = false; chatSendBtn.disabled = false; mic.disabled = false; };
    const label = u.bubble.querySelector('.voice-text');
    try {
      const fd = new FormData();
      const t = blob.type || '';
      fd.append('audio', blob, 'voice.' + (t.includes('mp4') ? 'm4a' : t.includes('ogg') ? 'ogg' : 'webm'));
      fd.append('lang', sel.value);
      const res = await fetch('/api/voice', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.success) { label.textContent = ''; bot.bubble.textContent = data.error || 'Something went wrong.'; done(); return; }
      label.textContent = '“' + data.transcript + '”';
      const meta = document.createElement('small');
      meta.textContent = '  ·  ' + (data.language_name ? data.language_name + '  ·  ' : '') + data.stt_seconds + 's' +
                         (data.source === 'gemini' ? '  ·  double-checked' : '') + (data.uncertain ? '  ·  not sure' : '');
      label.appendChild(meta);
      revealText(bot.bubble, data.response, () => {
        done();
        if (speakBtn.classList.contains('on')) speak(data.response, data.language);
      });
    } catch (e) {
      label.textContent = ''; bot.bubble.textContent = 'Error: ' + e.message; done();
    }
  }
})();
"""

_voice2_prev_get_html = WebUI.get_html


def _voice2_get_html(self) -> str:
    html = _voice2_prev_get_html(self)
    html = html.replace('</head>', '<style>' + VOICE2_CSS + '</style></head>', 1)
    html = html.replace('</body>', '<script>' + VOICE2_JS + '</script></body>', 1)
    return html


WebUI.get_html = _voice2_get_html

# ============================================================================
# LONG-JOB FIX: fake news + voice run in the background and the browser polls.
# (Cloudflare quick tunnels kill any single request longer than ~100 s and
#  return an HTML page, which caused "Unexpected token '<'".)
# ============================================================================
JOBS = {}


def _job_start(fn):
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["t"] > 1800]:
        JOBS.pop(k, None)
    jid = uuid.uuid4().hex
    job = {"done": False, "result": None, "status": 200, "t": now}
    JOBS[jid] = job

    def run():
        try:
            job["result"] = fn()
        except ValueError as e:
            job["result"], job["status"] = {"success": False, "error": str(e)}, 400
        except Exception as e:
            print("Job error:", e)
            job["result"], job["status"] = {"success": False, "error": str(e)}, 500
        job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"success": True, "job": jid})


def _jobs_add_routes(self):
    @self.app.route('/api/job/<jid>')
    def job_status(jid):
        job = JOBS.get(jid)
        if not job:
            return jsonify({"done": True, "status": 404,
                            "result": {"success": False, "error": "Unknown or expired job."}})
        if not job["done"]:
            return jsonify({"done": False})
        return jsonify({"done": True, "status": job["status"], "result": job["result"]})

    @self.app.route('/api/news_start', methods=['POST'])
    def news_start():
        text = (request.json or {}).get('text', '')
        if not text.strip():
            return jsonify({'success': False, 'error': 'Please enter some text.'}), 400
        return _job_start(lambda: self.fake_news_detector.detect(text))

    @self.app.route('/api/voice_start', methods=['POST'])
    def voice_start():
        f = request.files.get('audio')
        if not f:
            return jsonify({'success': False, 'error': 'No audio received.'}), 400
        lang_pref = (request.form.get('lang') or 'auto').strip().lower()
        mt = (f.mimetype or '').lower()
        suffix = '.ogg' if 'ogg' in mt else '.m4a' if ('mp4' in mt or 'aac' in mt) else '.wav' if 'wav' in mt else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp)
            path = tmp.name

        def work():
            try:
                t0 = time.time()
                res = self.voice.transcribe(path, lang_pref, self.chatbot)
                stt = time.time() - t0
                _VOICE_TL.on = True
                try:
                    reply = self.chatbot.chat(res["text"])
                finally:
                    _VOICE_TL.on = False
                return {'success': True, 'transcript': res["text"], 'language': res["language"],
                        'language_name': LANG_NAMES.get(res["language"], res["language"] or ''),
                        'source': res["source"], 'uncertain': res["uncertain"],
                        'stt_seconds': round(stt, 1), 'response': reply}
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return _job_start(work)


_jobs_prev_init = WebUI.__init__


def _jobs_init(self):
    _jobs_prev_init(self)
    if FLASK_AVAILABLE:
        _jobs_add_routes(self)


WebUI.__init__ = _jobs_init

# Browser side: silently turn the two slow calls into start + poll.
JOBS_JS = r"""
(function () {
  const orig = window.fetch.bind(window);
  const MAP = { '/api/voice': '/api/voice_start', '/api/detect_fake_news': '/api/news_start' };
  const json = (obj, status) => new Response(JSON.stringify(obj), { status: status || 200, headers: { 'Content-Type': 'application/json' } });

  window.fetch = async function (url, opts) {
    const target = typeof url === 'string' ? MAP[url] : null;
    if (!target) return orig(url, opts);

    const r = await orig(target, opts);
    const txt = await r.text();
    let start;
    try { start = JSON.parse(txt); }
    catch (e) { throw new Error('Server answered with a web page instead of data (HTTP ' + r.status + '). The tunnel may have dropped. Reload and try again.'); }
    if (!start.job) return json(start, r.status);

    let fails = 0;
    while (true) {
      await new Promise(s => setTimeout(s, 1000));
      try {
        const pr = await orig('/api/job/' + start.job);
        const p = JSON.parse(await pr.text());
        fails = 0;
        if (p.done) return json(p.result, p.status);
      } catch (e) {
        if (++fails > 90) throw new Error('Lost connection to the server.');
      }
    }
  };
})();
"""

_jobs_prev_get_html = WebUI.get_html


def _jobs_get_html(self) -> str:
    html = _jobs_prev_get_html(self)
    return html.replace('</body>', '<script>' + JOBS_JS + '</script></body>', 1)


WebUI.get_html = _jobs_get_html

# ============================================================================
# MIC FIX: no browser noise filters, microphone picker, live level test.
# ============================================================================
# 1) Record without the browser filters that can silence a voice.
VOICE2_JS = VOICE2_JS.replace(
    "audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }",
    "audio: window.__pipAudioConstraints()",
)

MIC_JS = r"""
(function () {
  let micId = '';
  try { micId = localStorage.getItem('pipMicId') || ''; } catch (e) {}

  // used by the recorder every time it starts
  window.__pipAudioConstraints = function () {
    const c = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true };
    if (micId) c.deviceId = { ideal: micId };
    return c;
  };

  const box = document.querySelector('#chat .chat-head .mic-btn.mini');
  if (!box) return;
  const wrap = box.parentNode;

  const sel = document.createElement('select');
  sel.id = 'mic-device';
  sel.title = 'Which microphone to use';
  sel.style.cssText = 'width:auto;max-width:170px;padding:6px 10px;font-size:14px;border-radius:12px';
  sel.innerHTML = '<option value="">Default microphone</option>';

  const test = document.createElement('button');
  test.type = 'button'; test.className = 'mic-btn mini'; test.textContent = '🎚';
  test.title = 'Test the microphone (speak for a few seconds)';

  const meter = document.createElement('div');
  meter.style.cssText = 'display:none;align-items:center;gap:8px;font-size:13px';
  meter.innerHTML = '<div style="width:90px;height:10px;border-radius:6px;background:rgba(255,255,255,.2);overflow:hidden">' +
                    '<i id="mic-bar" style="display:block;height:100%;width:0;background:#22c55e"></i></div>' +
                    '<span id="mic-db">…</span>';

  wrap.insertBefore(sel, wrap.firstChild);
  wrap.insertBefore(test, wrap.firstChild);
  wrap.insertBefore(meter, wrap.firstChild);

  async function listDevices() {
    try {
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
      sel.innerHTML = '<option value="">Default microphone</option>' +
        devs.map((d, i) => '<option value="' + d.deviceId + '">' + (d.label || 'Microphone ' + (i + 1)) + '</option>').join('');
      sel.value = micId;
      if (sel.value !== micId) sel.value = '';
    } catch (e) {}
  }
  listDevices();
  if (navigator.mediaDevices) navigator.mediaDevices.addEventListener('devicechange', listDevices);

  sel.addEventListener('change', () => {
    micId = sel.value;
    try { localStorage.setItem('pipMicId', micId); } catch (e) {}
  });

  let testing = false;
  test.addEventListener('click', async () => {
    if (testing) return;
    testing = true; test.disabled = true;
    const bar = document.getElementById('mic-bar'), db = document.getElementById('mic-db');
    meter.style.display = 'flex';
    let stream, ctx;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: window.__pipAudioConstraints() });
      await listDevices();                                   // names appear after permission is given
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      const an = ctx.createAnalyser(); an.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(an);
      const buf = new Float32Array(an.fftSize);
      let peak = -120;
      const t0 = Date.now();
      await new Promise(done => {
        const iv = setInterval(() => {
          an.getFloatTimeDomainData(buf);
          let m = 0; for (let i = 0; i < buf.length; i++) m = Math.max(m, Math.abs(buf[i]));
          const d = 20 * Math.log10(m + 1e-9);
          peak = Math.max(peak, d);
          bar.style.width = Math.max(0, Math.min(100, (d + 70) * 100 / 60)) + '%';
          db.textContent = 'Say something… ' + Math.round(d) + ' dB';
          if (Date.now() - t0 > 6000) { clearInterval(iv); done(); }
        }, 50);
      });
      db.textContent = peak > -45 ? '✅ Mic works (peak ' + Math.round(peak) + ' dB)'
                                  : '⚠️ Silent (peak ' + Math.round(peak) + ' dB). Pick another microphone.';
    } catch (e) {
      db.textContent = '❌ ' + (e.name === 'NotAllowedError' ? 'Microphone blocked. Allow it in the address bar.' : e.message);
    } finally {
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (ctx) { try { ctx.close(); } catch (e) {} }
      testing = false; test.disabled = false;
    }
  });
})();
"""

_mic_prev_get_html = WebUI.get_html


def _mic_get_html(self) -> str:
    html = _mic_prev_get_html(self)
    return html.replace('</body>', '<script>' + MIC_JS + '</script></body>', 1)


WebUI.get_html = _mic_get_html

# ============================================================================
# QUIET-MIC FIX: software boost + compressor in the browser, and a much
# lower silence gate on the server, so soft voices on weak laptop mics work.
# ============================================================================

BOOST_JS = r"""
(function () {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
  let boost = 20;
  try { boost = parseFloat(localStorage.getItem('pipMicBoost')) || 20; } catch (e) {}

  const origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async function (c) {
    const raw = await origGUM(c);
    if (!c || !c.audio || c.video) return raw;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return raw;
    try {
      const ctx = new AC();
      if (ctx.state === 'suspended') ctx.resume();
      const src = ctx.createMediaStreamSource(raw);
      const gain = ctx.createGain(); gain.gain.value = boost;
      const comp = ctx.createDynamicsCompressor();      // stops loud parts from clipping
      comp.threshold.value = -18; comp.knee.value = 12; comp.ratio.value = 8;
      comp.attack.value = 0.003; comp.release.value = 0.2;
      const dest = ctx.createMediaStreamDestination();
      src.connect(gain); gain.connect(comp); comp.connect(dest);
      const out = dest.stream;
      out.getTracks().forEach(t => {
        const stop = t.stop.bind(t);
        t.stop = () => { stop(); raw.getTracks().forEach(r => r.stop()); try { ctx.close(); } catch (e) {} };
      });
      return out;
    } catch (e) {
      return raw;                                        // if anything fails, use the plain mic
    }
  };

  const old = document.querySelector('#chat .chat-head #mic-device');
  const host = old ? old.parentNode : document.querySelector('#chat .chat-head');
  if (!host) return;
  const sel = document.createElement('select');
  sel.title = 'Software mic boost. Raise it if your voice is too quiet.';
  sel.style.cssText = 'width:auto;padding:6px 10px;font-size:14px;border-radius:12px';
  [1, 4, 10, 20, 40].forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = 'Boost ×' + v;
    sel.appendChild(o);
  });
  sel.value = String(boost);
  if (sel.value !== String(boost)) sel.value = '20';
  sel.addEventListener('change', () => {
    boost = parseFloat(sel.value);
    try { localStorage.setItem('pipMicBoost', String(boost)); } catch (e) {}
  });
  host.insertBefore(sel, host.firstChild);
})();
"""


# ============================================================================
# VOICE v3 FIX: honours the language dropdown (no forced English), gentler
# noise handling, Gemini listens first for weaker languages, and a debug
# endpoint to hear what the server received.
# ============================================================================
LAST_VOICE = {"wav": None}
WEAK_WHISPER_LANGS = {"hy", "ka", "az", "kk", "uz", "tg", "mn", "ps", "sd", "my", "lo", "km"}

# lighter compressor in the browser, so noise is not pumped up as loud as the voice
BOOST_JS = BOOST_JS.replace(
    "comp.threshold.value = -18; comp.knee.value = 12; comp.ratio.value = 8;",
    "comp.threshold.value = -30; comp.knee.value = 20; comp.ratio.value = 3;")

_v3_prev_init = WebUI.__init__

def _v3_init(self):
    _v3_prev_init(self)
    if FLASK_AVAILABLE:
        from flask import Response

        @self.app.route('/api/last_voice')
        def last_voice():
            if not LAST_VOICE["wav"]:
                return "No voice message received yet.", 404
            return Response(LAST_VOICE["wav"], mimetype="audio/wav")


WebUI.__init__ = _v3_init

# ============================================================================
# STOP BUTTONS + MIC STAYS ON
#  * auto-send on pause is OFF by default (recording ends only when you press stop)
#  * ✖ cancel button while recording
#  * ⏹ Stop buttons for chat, fake news, image analysis, code checker, calculator
# ============================================================================

# 1) auto-send off by default (new storage key so the old saved choice is ignored)
VOICE2_JS = VOICE2_JS.replace("localStorage.getItem('pipVoiceAuto') !== '0'",
                              "localStorage.getItem('pipVoiceAuto2') === '1'")
VOICE2_JS = VOICE2_JS.replace("localStorage.setItem('pipVoiceAuto',",
                              "localStorage.setItem('pipVoiceAuto2',")
VOICE2_JS = VOICE2_JS.replace("'Auto-send when I stop talking'",
                              "'Auto-send when I pause (off = you press the stop button to send)'")

# 2) let the polling loop notice a stop
JOBS_JS = JOBS_JS.replace(
    "await new Promise(s => setTimeout(s, 1000));",
    "await new Promise(s => setTimeout(s, 1000));\n"
    "      if (opts && opts.signal && opts.signal.aborted) throw new DOMException('Stopped', 'AbortError');",
    1)

STOP_CSS = r"""
.stop-btn{display:none;align-items:center;gap:6px;padding:9px 18px;border-radius:24px;cursor:pointer;
  font-weight:700;font-size:15px;color:#fff;background:#d92d2d;border:2px solid #ff8a8a;
  box-shadow:0 0 14px rgba(255,60,60,.55);margin-top:12px;transition:transform .15s}
.stop-btn:hover{transform:scale(1.06);background:#f03a3a}
.composer .stop-btn,.button-group .stop-btn{margin-top:0}
.mic-btn.cancel{display:none;background:rgba(0,0,0,.55);color:#ff9a9a;border-color:#ff8a8a}
"""

STOP_JS = r"""
(function () {
  const STOPPED = '⏹ Stopped.';
  const isStatus = u => u.indexOf('/api/analyze_status/') === 0;
  const groupOf = u => (u === '/api/chat' || u === '/api/voice') ? 'chat'
    : u === '/api/detect_fake_news' ? 'news'
    : (u === '/api/analyze_start' || isStatus(u)) ? 'image'
    : u === '/api/check_code' ? 'code'
    : u === '/api/calculate' ? 'calc' : null;

  const state = {}, btns = {};
  let revealStop = null;
  const show = (g, on) => { if (btns[g]) btns[g].style.display = on ? 'inline-flex' : 'none'; };
  const fake = o => new Response(JSON.stringify(o), { status: 200, headers: { 'Content-Type': 'application/json' } });
  const stoppedBody = u => u === '/api/chat' ? { success: true, response: STOPPED }
    : isStatus(u) ? { success: true, done: true, stage: 'Done', pct: 100, result: { success: false, error: STOPPED } }
    : { success: false, error: STOPPED };

  // ---------- wrap fetch: every request gets an abort controller ----------
  const base = window.fetch.bind(window);
  window.fetch = async function (url, opts) {
    const g = typeof url === 'string' ? groupOf(url) : null;
    if (!g) return base(url, opts);
    const s = state[g] = state[g] || { ctl: null, stopped: false };
    const status = isStatus(url);
    const o = Object.assign({}, opts || {});
    if (!status) { s.stopped = false; s.ctl = new AbortController(); o.signal = s.ctl.signal; show(g, true); }
    else if (s.stopped) { show(g, false); return fake(stoppedBody(url)); }
    try {
      const r = await base(url, o);
      if (status || g === 'image') {
        try {
          const j = await r.clone().json();
          if (status ? j.done : !j.success) show(g, false);
        } catch (e) {}
      } else if (g !== 'chat') {
        show(g, false);
      }
      return r;
    } catch (e) {
      show(g, false);
      if (s.stopped || e.name === 'AbortError') return fake(stoppedBody(url));
      throw e;
    }
  };

  function stop(g) {
    const s = state[g] = state[g] || { ctl: null, stopped: false };
    s.stopped = true;
    if (s.ctl) s.ctl.abort();
    if (g === 'chat') {
      if (window.speechSynthesis) speechSynthesis.cancel();
      if (revealStop) revealStop();
    }
    show(g, false);
  }

  // ---------- chat: the typing animation can be stopped too ----------
  window.revealText = function (bubble, full, done) {
    const step = Math.max(2, Math.ceil(full.length / 120));
    let i = 0, t = null;
    show('chat', true);
    const finish = () => { clearInterval(t); revealStop = null; show('chat', false); done(); };
    revealStop = () => { bubble.innerHTML = renderMarkdown(full.slice(0, i)) + '<br><em>' + STOPPED + '</em>'; finish(); };
    t = setInterval(() => {
      i = Math.min(full.length, i + step);
      bubble.innerHTML = renderMarkdown(full.slice(0, i));
      scrollChat();
      if (i >= full.length) finish();
    }, 16);
  };

  // ---------- build the buttons ----------
  function make(g, place) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'stop-btn'; b.textContent = '⏹ Stop'; b.title = 'Stop this';
    b.addEventListener('click', () => stop(g));
    place(b);
    btns[g] = b;
  }
  const after = sel => b => { const t = document.querySelector(sel); if (t) t.insertAdjacentElement('afterend', b); };

  make('chat', b => { const send = document.getElementById('chat-send'); send.parentNode.insertBefore(b, send); });
  make('news', after('#news .generate-button'));
  make('image', after('#image .generate-button'));
  make('code', after('#code .generate-button'));
  make('calc', b => { const bg = document.querySelector('#calculator .button-group'); if (bg) bg.appendChild(b); });

  // chat button also shows whenever the chat is busy (covers the "Transcribing" step)
  const send = document.getElementById('chat-send');
  new MutationObserver(() => { if (send.disabled) show('chat', true); else if (!revealStop) show('chat', false); })
    .observe(send, { attributes: true, attributeFilter: ['disabled'] });

  // ---------- cancel button while recording ----------
  const mic = document.querySelector('#chat .composer .mic-btn');
  if (mic) {
    const x = document.createElement('button');
    x.type = 'button'; x.className = 'mic-btn cancel'; x.textContent = '✖'; x.title = 'Cancel recording (nothing is sent)';
    mic.parentNode.insertBefore(x, mic);
    x.addEventListener('click', () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' })));
    new MutationObserver(() => { x.style.display = mic.classList.contains('rec') ? 'inline-block' : 'none'; })
      .observe(mic, { attributes: true, attributeFilter: ['class'] });
  }
})();
"""

_stop_prev_get_html = WebUI.get_html


def _stop_get_html(self) -> str:
    html = _stop_prev_get_html(self)
    html = html.replace('</head>', '<style>' + STOP_CSS + '</style></head>', 1)
    return html.replace('</body>', '<script>' + STOP_JS + '</script></body>', 1)


WebUI.get_html = _stop_get_html

# ============================================================================
# MUTE THE STORM SOUND (and read-aloud) WHILE THE MIC IS RECORDING
# ============================================================================
# expose a mute switch from inside the storm script (it owns the audio nodes)
STORM_JS = STORM_JS.replace(
    "  // ---------- input ----------",
    "  window.__pipStormMute = function (m) {\n"
    "    try { if (master && audio) master.gain.setTargetAtTime(m ? 0 : .9, audio.currentTime, .02); } catch (e) {}\n"
    "  };\n"
    "  // ---------- input ----------", 1)

QUIET_JS = r"""
(function () {
  const mic = document.querySelector('#chat .composer .mic-btn:not(.cancel)');
  if (!mic) return;
  let was = false;
  new MutationObserver(() => {
    const on = mic.classList.contains('rec');
    if (on === was) return;
    was = on;
    if (on && window.speechSynthesis) speechSynthesis.cancel();   // read-aloud would be recorded too
    if (window.__pipStormMute) window.__pipStormMute(on);         // silence rain and thunder while recording
  }).observe(mic, { attributes: true, attributeFilter: ['class'] });
})();
"""

_quiet_prev_get_html = WebUI.get_html


def _quiet_get_html(self) -> str:
    html = _quiet_prev_get_html(self)
    return html.replace('</body>', '<script>' + QUIET_JS + '</script></body>', 1)


WebUI.get_html = _quiet_get_html

# ============================================================================
# REAL MUTE FOR THE STORM SOUND + HINT IN THE VOICE ERROR
# ============================================================================
STORM_JS = STORM_JS.replace(
    "  // ---------- input ----------",
    """  sbtn.addEventListener('click', () => {
    if (!audio) return;
    if (soundOn) { master.gain.setValueAtTime(.9, audio.currentTime); audio.resume(); }
    else { audio.suspend(); }
  });
  window.__pipSound = {
    isOn: () => soundOn,
    off: () => {
      if (!soundOn) return;
      soundOn = false; sbtn.textContent = '🔇'; sbtn.classList.remove('on');
      if (audio) { applyRainVolume(); audio.suspend(); }
    }
  };
  // ---------- input ----------""", 1)

VOICE2_JS = VOICE2_JS.replace(
    "bot.bubble.textContent = data.error || 'Something went wrong.';",
    "bot.bubble.textContent = (data.error || 'Something went wrong.') + "
    "(window.__pipSound && window.__pipSound.isOn() ? "
    "'  🔊 The storm sound is on, so the microphone may be hearing it instead of you. "
    "Click the 🔊 button to mute it, then try again.' : '');", 1)

QUIET_JS = r"""
(function () {
  const mic = document.querySelector('#chat .composer .mic-btn:not(.cancel)');
  if (!mic) return;
  let was = false;
  new MutationObserver(() => {
    const on = mic.classList.contains('rec');
    if (on === was) return;
    was = on;
    if (on) {
      if (window.speechSynthesis) speechSynthesis.cancel();   // read-aloud would be recorded too
      if (window.__pipSound) window.__pipSound.off();         // storm sound fully off until you press 🔊
    }
  }).observe(mic, { attributes: true, attributeFilter: ['class'] });
})();
"""

# ============================================================================
# REMOVE THE STORM SOUND (button + audio). Visuals stay.
# ============================================================================
STORM_CSS += "\n#stormSound{display:none!important}\n"
STORM_JS = STORM_JS.replace("document.body.appendChild(sbtn);", "")
STORM_JS = STORM_JS.replace("function thunder(d01) {", "function thunder(d01) {\n    return;", 1)

# ============================================================================
# SIMPLE VOICE: raw audio, Gemini listens first, Whisper is the backup
# ============================================================================
os.environ["VOICE_DENOISE"] = "0"
BOOST_JS = BOOST_JS.replace("localStorage.getItem('pipMicBoost')) || 20;",
                            "localStorage.getItem('pipMicBoost1')) || 1;")
BOOST_JS = BOOST_JS.replace("localStorage.setItem('pipMicBoost',", "localStorage.setItem('pipMicBoost1',")
BOOST_JS = BOOST_JS.replace("sel.value = '20';", "sel.value = '1';")


def _clean_plain(self, a):
    dur = a.size / self.SR
    stats = {"seconds": round(dur, 2), "peak_db": -120.0, "rms_db": -120.0}
    if a.size < self.SR * 0.25:
        return None, 0.0, stats
    a = a - float(a.mean())
    peak = float(np.max(np.abs(a)))
    stats["peak_db"] = round(20 * np.log10(peak + 1e-9), 1)
    stats["rms_db"] = round(20 * np.log10(float(np.sqrt(np.mean(a ** 2) + 1e-12)) + 1e-9), 1)
    if peak < 0.0004:
        return None, 0.0, stats
    a = np.clip(a * min(10.0, 0.9 / peak), -1, 1).astype(np.float32)
    return a, dur, stats


def _transcribe_gemini_first(self, path, lang_pref="auto", bot=None):
    audio = self._decode(path)
    audio, speech_s, st = self._clean(audio)
    print(f"Voice input: {st['seconds']}s, peak {st['peak_db']} dB")
    if audio is None:
        raise ValueError("The microphone sent almost no sound. Check the microphone and try again.")
    LAST_VOICE["wav"] = _wav_bytes(audio)

    pick = lang_pref if lang_pref not in ("", "auto") else (os.getenv("VOICE_LANGUAGE") or "").strip().lower()
    lang = pick or None
    secs = round(audio.size / self.SR, 1)

    if bot is not None:
        try:
            g = bot.transcribe_audio(_wav_bytes(audio), LANG_NAMES.get(lang, lang) if lang else None)
            if g and not self._suspicious(g, speech_s):
                return {"text": g, "language": lang or "", "lang_confidence": 1.0,
                        "source": "gemini", "uncertain": False, "seconds": secs}
            print("Gemini heard:", repr(g))
        except Exception as e:
            print("Gemini audio failed:", str(e)[:150])

    text, reason = "", None
    with self._lock:
        if self._load():
            feats = self._features(self._chunks(audio))
            if not lang:
                lang, _ = self._pick_lang(self._detect(feats))
            text = self._run(feats, lang, 5)
            reason = self._suspicious(text, speech_s)
    print(f"Whisper ({lang}) heard: {text!r} | problem: {reason}")
    if not text or reason in ("empty", "hallucination"):
        raise ValueError("I couldn't make out what you said. Open /api/last_voice to hear what the server received.")
    return {"text": text, "language": lang or "", "lang_confidence": 1.0,
            "source": "whisper", "uncertain": bool(reason), "seconds": secs}


VoiceTranscriber._clean = _clean_plain
VoiceTranscriber.transcribe = _transcribe_gemini_first

MIC_JS = MIC_JS.replace(
    "echoCancellation: false, noiseSuppression: false, autoGainControl: true",
    "echoCancellation: true, noiseSuppression: true, autoGainControl: true")

# ============================================================================
# RESPONSIVE + CROSS-BROWSER PATCH
# Paste this whole block ABOVE the line:   if __name__ == "__main__":
#
#  * Phones, tablets, laptops, ultrawide screens (and rotation)
#  * Safe areas (iPhone notch), 100dvh (mobile address bar), 16px inputs (no iOS zoom)
#  * Sidebar gets a dark backdrop on small screens; tap outside or press Esc to close
#  * Voice settings fold behind a gear button on phones
#  * Fallbacks for Safari / older Firefox (animated rainbow border, SVG filters,
#    backdrop-filter prefix, tap/hover effects that stick on touch screens)
#  * Effects switch: automatic "lite" mode on weak phones / reduced-motion,
#    manual toggle in the sidebar, or add ?lite=1 / ?lite=0 to the link
#  * Hides the mic button if the browser cannot record
# ============================================================================

RESP_META = (
    '<meta name="theme-color" content="#05070d">'
    '<meta name="color-scheme" content="dark light">'
    '<meta name="mobile-web-app-capable" content="yes">'
    '<meta name="apple-mobile-web-app-capable" content="yes">'
    '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
)

RESP_CSS = r"""
:root{--safe-t:env(safe-area-inset-top,0px);--safe-b:env(safe-area-inset-bottom,0px);
  --safe-l:env(safe-area-inset-left,0px);--safe-r:env(safe-area-inset-right,0px)}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{-webkit-tap-highlight-color:transparent;overscroll-behavior-y:none;word-wrap:break-word}
img,video,audio{max-width:100%}
button,select,.particle-btn,.generate-button{touch-action:manipulation}

/* Safari needs the -webkit- prefix */
.tab-inner,.sidebar{-webkit-backdrop-filter:blur(10px)}
body.light .tab-inner{-webkit-backdrop-filter:blur(14px)}

/* Browsers without @property: static (not spinning) rainbow border */
html.no-at-property .tab-content::before{
  background:conic-gradient(from 0deg,hsl(0 100% 62%),hsl(45 100% 60%),hsl(100 85% 55%),hsl(175 90% 55%),
  hsl(225 100% 65%),hsl(275 100% 68%),hsl(325 100% 65%),hsl(360 100% 62%));animation:none}
/* SVG filters on normal HTML are slow or ignored in Safari */
html.safari .tab-content::before{filter:none}

/* sidebar backdrop (small screens) */
#sideScrim{position:fixed;inset:0;z-index:19;background:rgba(0,0,0,.55);opacity:0;pointer-events:none;transition:opacity .3s}
body.sidebar-open #sideScrim{opacity:1;pointer-events:auto}
@media (min-width:900px){#sideScrim{display:none}}

/* effects switch button (inside the sidebar) */
.lite-btn{margin-top:6px;padding:12px 18px;border-radius:100px;cursor:pointer;font-size:14px;width:100%;text-align:left;
  color:#ffaa5c;background:rgba(0,0,0,.6);border:2px solid rgba(221,132,72,.5)}
.lite-btn:hover{background:rgba(221,132,72,.25);color:#fff}
body.light .lite-btn{color:#2f6b9a;background:rgba(255,255,255,.6);border-color:#bfe1f5}

/* voice settings: gear button is only shown on phones */
.gear-btn{display:none}

/* very wide screens */
@media (min-width:1600px){.container{max-width:1280px}}

/* tablets and phones: sidebar slides over the page */
@media (max-width:899px){
  .sidebar{width:min(270px,84vw);padding-top:calc(112px + var(--safe-t));padding-left:calc(16px + var(--safe-l))}
}

/* phones */
@media (max-width:700px){
  #Button{width:118px;height:88px;left:2px;top:var(--safe-t)}
  .theme-wrap{top:calc(8px + var(--safe-t));right:calc(12px + var(--safe-r));--scale:.9}
  .theme-label{font-size:18px}
  .container{padding:calc(88px + var(--safe-t)) calc(12px + var(--safe-r)) calc(40px + var(--safe-b)) calc(12px + var(--safe-l))}
  .tab-inner{padding:16px!important}
  h2{font-size:1.25em;margin-bottom:14px}
  body.light h2,body.light .chat-welcome h3{font-size:1.9em}
  .chat-welcome h3{font-size:1.5em}
  input[type="text"],textarea,select{font-size:16px!important}
  .button-group .generate-button{flex:1 1 140px}
  #image .field-row{grid-template-columns:1fr!important}
  #code-input{min-height:220px}
  #news-input{min-height:180px}
  .output{padding:14px;overflow-x:auto}
  #img-output table,#news-output table{display:block;overflow-x:auto}

  /* chat fills the screen and shrinks with the mobile address bar */
  #chat .tab-inner{height:calc(100vh - 150px)!important;height:calc(100dvh - 150px - var(--safe-t) - var(--safe-b))!important;
    min-height:380px;padding:12px!important}
  .chat-container{padding:16px 12px}
  .chat-head{flex-wrap:wrap;gap:8px}
  .chat-head>div{flex-wrap:wrap;justify-content:flex-end}
  .chat-head.show-opts>div{width:100%;justify-content:flex-start}
  .gear-btn{display:inline-block}
  .chat-head:not(.show-opts) .vopt{display:none!important}
  .composer{gap:6px;padding:6px 6px 6px 12px}
  .composer .mic-btn{width:42px;height:42px}
  .composer .generate-button svg.icon{display:none}
  .composer .generate-button.compact{padding:10px 16px}
  .chips .generate-button{min-width:0}
  .msg-row{gap:8px}
  .avatar{width:30px;height:30px}
  .tab-content::before{filter:none}
  .stop-btn{padding:8px 14px;font-size:14px}
}

/* very small phones */
@media (max-width:360px){
  .container{padding-left:8px;padding-right:8px}
  .generate-button{min-width:0}
}

/* touch screens: hover effects would stay stuck after a tap */
@media (hover:none){
  .particle-btn:hover{transform:none}
  .generate-button:hover{--generate-button-scale:1}
  #Button:hover{transform:none}
}

/* people who asked their device for less motion */
@media (prefers-reduced-motion:reduce){
  .tab-content::before,.pip-cw,.pip-sun,.pip-sun:before,#stars,#stars2,#stars3,.generate-button:after{animation:none!important}
}

/* lite mode: no storm, no galaxy, no blur = smooth on weak devices */
body.lite #storm3d,body.lite #stars-wrap,body.lite .panel-stars,body.lite #stormy-canvas{display:none!important}
body.lite:not(.light){background:radial-gradient(ellipse at 50% 100%,#1b2740 0%,#05070d 70%) fixed}
body.lite .tab-content::before{filter:none;animation:none}
body.lite .tab-inner,body.lite .sidebar{backdrop-filter:none!important;-webkit-backdrop-filter:none!important}
body.lite .pip-cloud{filter:none}
body.lite .pip-sun:before{display:none}
/* lite mode in light theme: same treatment — flat sky, no sun, no clouds, no petals */
body.lite.light #pipSky,body.lite.light #petalCanvas{display:none!important}
body.lite.light{background:linear-gradient(180deg,#2c8be6 0%,#5ab9ff 55%,#c4e8ff 100%) fixed}
"""

RESP_JS = r"""
(function () {
  const root = document.documentElement;

  // ---------- browser feature checks ----------
  if (!(window.CSS && CSS.registerProperty)) root.classList.add('no-at-property');
  if (/^((?!chrome|android|crios).)*safari/i.test(navigator.userAgent) || /iP(hone|ad|od)/.test(navigator.userAgent)) {
    root.classList.add('safari');
  }

  // ---------- lite mode (auto, saved, or ?lite=1 / ?lite=0) ----------
  let lite = false;
  try {
    const forced = new URLSearchParams(location.search).get('lite');
    let saved = null;
    try { saved = localStorage.getItem('pipLite'); } catch (e) {}
    if (forced !== null) {
      lite = forced === '1';
      try { localStorage.setItem('pipLite', lite ? '1' : '0'); } catch (e) {}
    } else if (saved !== null) {
      lite = saved === '1';
    } else {
      const reduced = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
      const touch = window.matchMedia && matchMedia('(pointer: coarse)').matches;
      const weak = (navigator.deviceMemory && navigator.deviceMemory <= 4) ||
                   (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4);
      lite = !!(reduced || (touch && weak));
    }
  } catch (e) {}
  document.body.classList.toggle('lite', lite);

  const tabs = document.querySelector('#sidebar .tabs');
  if (tabs) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'lite-btn';
    const label = () => { b.textContent = document.body.classList.contains('lite') ? '✨ Effects: off (faster)' : '✨ Effects: on'; };
    label();
    b.addEventListener('click', () => {
      const on = !document.body.classList.contains('lite');
      document.body.classList.toggle('lite', on);
      try { localStorage.setItem('pipLite', on ? '1' : '0'); } catch (e) {}
      label();
    });
    tabs.appendChild(b);
  }

  // ---------- sidebar: tap outside or press Esc to close ----------
  const scrim = document.createElement('div');
  scrim.id = 'sideScrim';
  document.body.appendChild(scrim);
  scrim.addEventListener('click', () => { if (typeof setSidebar === 'function') setSidebar(false); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && window.innerWidth < 900 && document.body.classList.contains('sidebar-open') &&
        typeof setSidebar === 'function') setSidebar(false);
  });

  // ---------- voice settings behind a gear button (phones) ----------
  const head = document.querySelector('#chat .chat-head');
  const wrap = head ? head.querySelector('div') : null;
  if (wrap) {
    Array.from(wrap.children).forEach(el => {
      if (el.classList.contains('generate-button')) return;                 // Clear
      if (el.classList.contains('mic-btn') && el.textContent.trim() === '🔊') return;   // read aloud
      el.classList.add('vopt');                                             // language, mic picker, test, auto-send
    });
    const gear = document.createElement('button');
    gear.type = 'button'; gear.className = 'mic-btn mini gear-btn'; gear.textContent = '⚙';
    gear.title = 'Voice settings';
    gear.addEventListener('click', () => { head.classList.toggle('show-opts'); gear.classList.toggle('on'); });
    wrap.insertBefore(gear, wrap.firstChild);
  }

  // ---------- small touch-keyboard helpers ----------
  const chatIn = document.getElementById('chat-input');
  if (chatIn) {
    chatIn.setAttribute('enterkeyhint', 'send');
    chatIn.setAttribute('autocomplete', 'off');
    chatIn.addEventListener('focus', () => {
      setTimeout(() => { try { chatIn.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (e) {} }, 350);
    });
  }
  ['calc-input', 'calc-variable', 'code-input'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.setAttribute('autocapitalize', 'off'); el.setAttribute('autocorrect', 'off'); el.setAttribute('spellcheck', 'false'); }
  });

  // ---------- browsers that cannot record ----------
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder)) {
    document.querySelectorAll('#chat .composer .mic-btn, #chat .chat-head .gear-btn').forEach(b => { b.style.display = 'none'; });
    if (chatIn) chatIn.placeholder = 'Message pip ai agent... (voice is not supported in this browser)';
  }

  // ---------- browser bar colour follows the theme ----------
  const meta = document.querySelector('meta[name="theme-color"]');
  const syncMeta = () => { if (meta) meta.setAttribute('content', document.body.classList.contains('light') ? '#5ab9ff' : '#05070d'); };
  syncMeta();
  new MutationObserver(syncMeta).observe(document.body, { attributes: true, attributeFilter: ['class'] });
})();
"""

# the storm and galaxy loops also stop while lite mode is on
STORM_JS = STORM_JS.replace(
    "if (light || document.hidden) { last = now; return; }",
    "if (light || document.hidden || document.body.classList.contains('lite')) { last = now; return; }")
GALAXY_JS = GALAXY_JS.replace(
    "if (document.hidden || document.body.classList.contains('light')) return;",
    "if (document.hidden || document.body.classList.contains('light') || document.body.classList.contains('lite')) return;")

_resp_prev_get_html = WebUI.get_html


def _resp_get_html(self) -> str:
    html = _resp_prev_get_html(self)
    html = html.replace('width=device-width, initial-scale=1.0',
                        'width=device-width, initial-scale=1.0, viewport-fit=cover', 1)
    html = html.replace('</head>', RESP_META + '<style>' + RESP_CSS + '</style></head>', 1)
    return html.replace('</body>', '<script>' + RESP_JS + '</script></body>', 1)


WebUI.get_html = _resp_get_html

# ============================================================================
# EFFECTS-ON-EVERY-DEVICE PATCH
# Paste this block BELOW the responsive patch and ABOVE: if __name__ == "__main__":
#
#  * The dark-mode storm, rain, lightning, galaxy and stars are ALWAYS on.
#    (The earlier "lite mode" turned them off by itself on phones, and also
#     when a phone had battery saver / "remove animations" switched on.)
#  * Instead of switching off, they get lighter on small screens:
#    fewer raindrops, lower pixel density, smaller galaxy picture, fewer petals.
#  * The "Effects" button in the sidebar still works if someone wants to turn
#    them off by hand. Old saved choices are ignored.
# ============================================================================

def _swap(text, old, new, name):
    if old not in text:
        print(f"⚠️ effects patch: could not find {name}, skipped")
        return text
    return text.replace(old, new, 1)


# 1) never switch lite mode on by itself, and ignore any old saved choice
RESP_JS = _swap(RESP_JS, "lite = !!(reduced || (touch && weak));", "lite = false;", "auto lite rule")
RESP_JS = RESP_JS.replace("pipLite", "pipLite2")

# 2) "reduce motion" must not freeze the sky, the stars or the border
RESP_CSS = re.sub(r"@media \(prefers-reduced-motion:reduce\)\{.*?\n\}\n", "", RESP_CSS, flags=re.S)

# 3) storm: fewer drops and lower pixel density on small screens
STORM_JS = _swap(
    STORM_JS,
    "Math.round(W * H / 620), 500, 1800",
    "Math.round(W * H / (W < 700 ? 1000 : 620)), 280, 1800",
    "raindrop count")
STORM_JS = _swap(
    STORM_JS,
    "dpr = Math.min(window.devicePixelRatio || 1, 1.5);",
    "dpr = Math.min(window.devicePixelRatio || 1, window.innerWidth < 700 ? 1.25 : 1.5);",
    "storm pixel density")

# 4) galaxy + stars: lower pixel density and a smaller galaxy picture on small screens
GALAXY_JS = _swap(
    GALAXY_JS,
    "const DPR = Math.min(window.devicePixelRatio || 1, 1.5);",
    "const DPR = Math.min(window.devicePixelRatio || 1, window.innerWidth < 700 ? 1.25 : 1.5);",
    "galaxy pixel density")
GALAXY_JS = _swap(
    GALAXY_JS,
    "const need = Math.min(2600,",
    "const need = Math.min(window.innerWidth < 700 ? 1300 : 2600,",
    "galaxy picture size")

# 5) light mode: fewer falling petals on small screens
PIP_JS = _swap(
    PIP_JS,
    "for (let i = 0; i < 80; i++) petals.push(new Petal());",
    "for (let i = 0; i < (window.innerWidth < 700 ? 40 : 80); i++) petals.push(new Petal());",
    "petal count")

# ============================================================================
# LOGO PATCH  (the "pip ai agent" button in the top-left corner)
# Paste this block ABOVE the line:   if __name__ == "__main__":
# (put it after the other patches)
#
#  DARK MODE : the blob becomes a tiny glowing ghost that floats, blinks and
#              holds the words "pip ai agent". When the mouse is on it (or it is
#              pressed, focused with the keyboard, or the sidebar is open) it
#              turns RED, frowns, grows angry eyebrows, raises its fists,
#              shakes and puffs steam.
#  LIGHT MODE: same blob as before, but glossy and see-through, in the pastel
#              pink / blue / yellow colours of the light theme, with a shine.
# ============================================================================

# states that count as "selected / angry"
_ANGRY_STATES = ['#Button:hover', '#Button:focus-visible', '#Button:active', '#Button[aria-expanded="true"]']


def _ang(sel):
    return ", ".join(f"{s} {sel}" for s in _ANGRY_STATES)


# ---------------------------------------------------------------- markup
LOGO_DEFS = r"""
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">
  <defs>
    <linearGradient id="pipFrontGloss" x1="0" y1="0" x2="0.35" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".92"/>
      <stop offset=".38" stop-color="#ffd0e4" stop-opacity=".55"/>
      <stop offset=".72" stop-color="#bfe6ff" stop-opacity=".5"/>
      <stop offset="1" stop-color="#fff0b0" stop-opacity=".62"/>
    </linearGradient>
    <linearGradient id="pipBackGloss" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ff9cc2" stop-opacity=".6"/>
      <stop offset=".55" stop-color="#9fd4ff" stop-opacity=".6"/>
      <stop offset="1" stop-color="#ffe08a" stop-opacity=".6"/>
    </linearGradient>
    <radialGradient id="pipShine" cx=".5" cy=".5" r=".5">
      <stop offset="0" stop-color="#ffffff" stop-opacity=".95"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </radialGradient>
  </defs>
</svg>
"""

LOGO_SHINE = r"""<g class="pip-shine" pointer-events="none">
          <ellipse cx="650" cy="448" rx="235" ry="48" transform="rotate(-9 650 448)" fill="url(#pipShine)"/>
          <ellipse cx="985" cy="528" rx="36" ry="21" transform="rotate(-25 985 528)" fill="#ffffff" opacity=".85"/>
          <ellipse cx="560" cy="655" rx="120" ry="16" transform="rotate(-6 560 655)" fill="#ffffff" opacity=".28"/>
        </g>
        """

GHOST_BODY = ("M10 56 C10 28 28 8 50 8 C72 8 90 28 90 56 L90 104 Q82 94 74 104 Q66 114 58 104 "
              "Q50 94 42 104 Q34 114 26 104 Q18 94 10 104 Z")

LOGO_GHOST = r"""<svg class="pip-ghost" viewBox="0 0 170 127" aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="pgVol" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#ffffff" stop-opacity=".5"/>
            <stop offset=".5" stop-color="#c4b5fd" stop-opacity=".1"/>
            <stop offset="1" stop-color="#4c1d95" stop-opacity=".3"/>
          </linearGradient>
          <linearGradient id="pgFade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#c4b5fd" stop-opacity=".8"/>
            <stop offset=".55" stop-color="#a78bfa" stop-opacity=".45"/>
            <stop offset="1" stop-color="#7c3aed" stop-opacity=".1"/>
          </linearGradient>
        </defs>
        <g class="g-shake">
          <g class="g-float">
            <g transform="translate(13 14) scale(.58)">
              <ellipse class="g-arm l" cx="6" cy="72" rx="9" ry="5.5"/>
              <ellipse class="g-arm r" cx="94" cy="72" rx="9" ry="5.5"/>
              <path class="g-body" d="__BODY__"/>
              <path class="g-shade" d="__BODY__" fill="url(#pgVol)"/>
              <ellipse class="g-cheek" cx="28" cy="63" rx="6.5" ry="3.8"/>
              <ellipse class="g-cheek" cx="72" cy="63" rx="6.5" ry="3.8"/>
              <g class="g-eyes"><g class="g-eyes-in">
                <ellipse class="eye" cx="37" cy="49" rx="6.5" ry="8.5"/>
                <ellipse class="eye" cx="63" cy="49" rx="6.5" ry="8.5"/>
                <circle class="g-hl" cx="39" cy="45.5" r="2.2"/>
                <circle class="g-hl" cx="65" cy="45.5" r="2.2"/>
              </g></g>
              <path class="g-brow" d="M25 35 L47 45"/>
              <path class="g-brow" d="M75 35 L53 45"/>
              <path class="g-smile" d="M42 65 Q50 72 58 65"/>
              <path class="g-grr" d="M36 77 L41 69 L46 77 L51 69 L56 77 L61 69 L64 75"/>
              <circle class="g-puff p1" cx="28" cy="6" r="5.5"/>
              <circle class="g-puff p2" cx="50" cy="-1" r="6.5"/>
              <circle class="g-puff p3" cx="72" cy="6" r="5.5"/>
              <path class="g-spark s1" transform="translate(97 14)" d="M-1 -7 H1 V-1 H7 V1 H1 V7 H-1 V1 H-7 V-1 H-1 Z"/>
              <path class="g-spark s2" transform="translate(5 36)" d="M-1 -6 H1 V-1 H6 V1 H1 V6 H-1 V1 H-6 V-1 H-1 Z"/>
              <path class="g-spark s3" transform="translate(101 72)" d="M-.8 -5 H.8 V-.8 H5 V.8 H.8 V5 H-.8 V.8 H-5 V-.8 H-.8 Z"/>
            </g>
          </g>
          <text class="g-text" x="84" y="57">pip ai</text>
          <text class="g-text" x="84" y="82">agent</text>
        </g>
      </svg>
      """.replace("__BODY__", GHOST_BODY)

# ----------------------------------------------------------------- style
LOGO_CSS = r"""
/* ---------- which logo shows ---------- */
body:not(.light) #Button .pip-blob{display:none}
body.light #Button .pip-ghost{display:none}

/* ================== DARK MODE: the ghost ================== */
.pip-ghost{display:block;width:100%;height:100%;overflow:visible;
  filter:drop-shadow(0 0 7px rgba(150,185,255,.8));transition:filter .35s}
.pip-ghost *{transform-box:fill-box;transform-origin:center}
.pip-ghost .g-body{fill:url(#pgFade);stroke:rgba(167,139,250,.85);stroke-width:2;transition:fill .35s,stroke .35s}
.pip-ghost .g-shade{fill:url(#pgVol);pointer-events:none}
.pip-ghost .g-arm{fill:rgba(196,181,253,.55);transition:fill .35s,transform .35s}
.pip-ghost .g-arm.l{transform:rotate(-25deg)}
.pip-ghost .g-arm.r{transform:rotate(25deg)}
.pip-ghost .g-cheek{fill:#ff9ec0;opacity:.4;transition:opacity .3s}
.pip-ghost .eye{fill:#2a1b4d;transition:fill .3s}
.pip-ghost .g-hl{fill:#fff;transition:opacity .3s}
.pip-ghost .g-eyes{animation:pipBlink 4.2s infinite}
.pip-ghost .g-eyes-in{transition:transform .3s}
.pip-ghost .g-brow{fill:none;stroke:#4a2a7a;stroke-width:2.6;stroke-linecap:round;opacity:0;transition:opacity .25s}
.pip-ghost .g-smile{fill:none;stroke:#2a1b4d;stroke-width:3.2;stroke-linecap:round;transition:opacity .25s}
.pip-ghost .g-grr{fill:none;stroke:#4a2a7a;stroke-width:3.2;stroke-linejoin:round;stroke-linecap:round;opacity:0;transition:opacity .25s}
.pip-ghost .g-puff{fill:#d8b4fe;opacity:0}
.pip-ghost .g-spark{fill:#fff;opacity:.9;animation:pipTw 2.8s ease-in-out infinite}
.pip-ghost .g-spark.s2{animation-delay:.9s}
.pip-ghost .g-spark.s3{animation-delay:1.7s}
.pip-ghost .g-float{animation:pipFloat 3.2s ease-in-out infinite}
.pip-ghost .g-text{fill:#d9d0ff;font-family:'Poppins','Segoe UI',Arial,sans-serif;font-weight:800;font-size:21px;
  paint-order:stroke;stroke:rgba(30,10,60,.65);stroke-width:3px;stroke-linejoin:round;transition:fill .35s}

@keyframes pipFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes pipBlink{0%,91%,100%{transform:scaleY(1)}94%{transform:scaleY(.08)}}
@keyframes pipTw{0%,100%{transform:scale(.55);opacity:.3}50%{transform:scale(1.2);opacity:1}}
@keyframes pipShake{0%,100%{transform:translate(0,0) rotate(0)}
  15%{transform:translate(-1.6px,.8px) rotate(-2deg)}30%{transform:translate(1.6px,-.8px) rotate(2deg)}
  45%{transform:translate(-1.2px,-.8px) rotate(-1.5deg)}60%{transform:translate(1.2px,.8px) rotate(1.5deg)}
  80%{transform:translate(-.8px,0) rotate(-1deg)}}
@keyframes pipSteam{0%{opacity:0;transform:translateY(8px) scale(.4)}25%{opacity:.95}
  100%{opacity:0;transform:translateY(-22px) scale(1.4)}}

/* ---- angry: mouse on it, pressed, keyboard focus, or sidebar open ---- */
__ANGRY_RULES__
@media (prefers-reduced-motion:reduce){.pip-ghost .g-shake{animation:none!important}}

/* ================== LIGHT MODE: glossy, see-through blob ================== */
body.light #Button .pip-blob{
  filter:drop-shadow(0 10px 14px rgba(255,120,170,.38)) drop-shadow(0 2px 3px rgba(255,255,255,.85))}
body.light #Button .pip-blob > rect{fill:url(#pipBackGloss)}
body.light #Button .Button__background{opacity:.3;mix-blend-mode:soft-light}
body.light #Button .frontBlob{fill:url(#pipFrontGloss);stroke:rgba(255,255,255,.95);stroke-width:7}
body.light #Button .Button__text{fill:#c2336f;paint-order:stroke;stroke:rgba(255,255,255,.9);stroke-width:8px;stroke-linejoin:round}
body.light #Button .pip-shine{animation:pipShimmer 3.4s ease-in-out infinite}
@keyframes pipShimmer{0%,100%{opacity:.8}50%{opacity:1}}
"""

_ANGRY_RULES = "\n".join([
    _ang(".pip-ghost") + "{filter:drop-shadow(0 0 11px rgba(168,85,247,.95))}",
    _ang(".pip-ghost .g-body") + "{fill:#a855f7;stroke:#e9d5ff}",
    _ang(".pip-ghost .g-shade") + "{opacity:.35}",
    _ang(".pip-ghost .g-arm") + "{fill:#c084fc}",
    _ang(".pip-ghost .g-arm.l") + "{transform:rotate(58deg)}",
    _ang(".pip-ghost .g-arm.r") + "{transform:rotate(-58deg)}",
    _ang(".pip-ghost .g-cheek") + "{opacity:0}",
    _ang(".pip-ghost .eye") + "{fill:#2a0a3d}",
    _ang(".pip-ghost .g-hl") + "{opacity:0}",
    _ang(".pip-ghost .g-eyes-in") + "{transform:scaleY(.62)}",
    _ang(".pip-ghost .g-brow") + "{opacity:1}",
    _ang(".pip-ghost .g-smile") + "{opacity:0}",
    _ang(".pip-ghost .g-grr") + "{opacity:1}",
    _ang(".pip-ghost .g-text") + "{fill:#e9d5ff;stroke:rgba(40,0,60,.7)}",
    _ang(".pip-ghost .g-puff") + "{animation:pipSteam 1s ease-out infinite}",
    _ang(".pip-ghost .g-puff.p2") + "{animation-delay:.3s}",
    _ang(".pip-ghost .g-puff.p3") + "{animation-delay:.6s}",
])
LOGO_CSS = LOGO_CSS.replace("__ANGRY_RULES__", _ANGRY_RULES)


# ------------------------------------------------------------ put it in
def _logo_swap(html, old, new, name):
    if old not in html:
        print(f"⚠️ logo patch: could not find {name}, skipped")
        return html
    return html.replace(old, new, 1)


_logo_prev_get_html = WebUI.get_html


def _logo_get_html(self) -> str:
    html = _logo_prev_get_html(self)
    html = _logo_swap(html, '<!-- Blob logo (top left)', LOGO_DEFS + '<!-- Blob logo (top left)', "logo defs")
    html = _logo_swap(html,
                      '<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true" viewBox="0 0 1490.01 1111.6">',
                      '<svg class="pip-blob" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" viewBox="0 0 1490.01 1111.6">',
                      "blob svg")
    html = _logo_swap(html, '<text class="Button__text"', LOGO_SHINE + '<text class="Button__text"', "blob text")
    html = _logo_swap(html, '<span class="screenreader-only">pip ai agent</span>',
                      LOGO_GHOST + '<span class="screenreader-only">pip ai agent</span>', "ghost slot")
    return html.replace('</head>', '<style>' + LOGO_CSS + '</style></head>', 1)
    

WebUI.get_html = _logo_get_html
# ============================================================================
# PART 2:  AUTO LANGUAGE  +  FACT-CHECK QUESTIONS/CLAIMS  +  SEARCHABLE LANGUAGE LISTS
#
#  * Fake News now accepts a question or claim too ("Are aliens real?",
#    "Is it true that ...?"). It is fact-checked with Google Search, not
#    against a pasted article. Long text and links still use the article check.
#    A "What to check" box lets you force one or the other.
#  * Auto language: the language is detected from what you wrote (or from your
#    browser for tabs without text) and the results are shown in it.
#  * Every language list is a search box: type "arm" or "հայ" and pick.
#  * The chatbot also verifies "is it true / is it real" questions with search.
# ============================================================================
def _target_lang(holder):
    """Language to translate a result into. None = leave it as it is."""
    code = _lang_code()
    if code != "auto":
        return code
    for c in (holder.get("detected_language"), getattr(_LANG_TL, "browser", "")):
        c = (str(c or "")).strip().lower().split("-")[0]
        if c in LANG_NAMES:
            return None if c == "en" else c
    return None


_ISO3_ALIAS = {"eng": "en", "hin": "hi", "spa": "es", "fre": "fr", "fra": "fr", "deu": "de",
               "ger": "de", "ara": "ar", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr",
               "urd": "ur", "fas": "fa", "per": "fa", "rus": "ru", "ukr": "uk", "jpn": "ja",
               "kor": "ko", "zho": "zh", "chi": "zh", "vie": "vi", "tha": "th", "ind": "id",
               "msa": "ms", "may": "ms", "tur": "tr", "ell": "el", "gre": "el", "heb": "he",
               "pan": "pa", "guj": "gu", "kan": "kn", "mal": "ml", "nep": "ne", "sin": "si",
               "mya": "my", "bur": "my", "khm": "km", "lao": "lo", "cat": "ca", "eus": "eu",
               "glg": "gl", "lat": "la", "swa": "sw", "amh": "am", "tir": "ti", "orm": "om"}


def _detect_language(bot, text):
    """ISO 639-1 code of the message's language. '' when unsure - callers default to English."""
    t = (text or "").strip().lower()
    # "hi"/"hello" etc. are English greetings - NOT Hindi (whose code is also "hi")
    if t in ("hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "yo", "sup",
             "hi!", "hello!", "hey!", "hi there", "hello there"):
        return "en"
    try:
        out = bot.plain("Reply with ONLY the ISO 639-1 language code (two lowercase letters) "
                        "of the main language of this text. IMPORTANT: the word 'hi' as a greeting "
                        "is English (en), not Hindi (hi).\n" + (text or "")[:600],
                        temperature=0.0, max_tokens=200)
        low = out.lower()
        names = {v.lower(): k for k, v in LANG_NAMES.items()}
        first = re.split(r"[^a-z]+", low.strip())[0] if low.strip() else ""
        for cand in (first, low.strip()):
            if cand in LANG_NAMES:
                return cand
            if cand in names:
                return names[cand]
            if cand in _ISO3_ALIAS:
                return _ISO3_ALIAS[cand]
        m = re.search(r"\b([a-z]{2,3})\b", low)
        if m:
            c = m.group(1)
            if c in LANG_NAMES:
                return c
            return _ISO3_ALIAS.get(c, c if len(c) == 2 else "")
    except Exception as e:
        print("Language detection skipped:", str(e)[:100])
    return ""


CLAIM_PROMPT = """You are a careful professional fact-checker. Today's date is __TODAY__.
The user wrote the text below. It may be a question ("Are aliens real?"), a claim, a headline or a rumor, in ANY language. Use Google Search to find out what reliable sources (scientific bodies, official agencies, news agencies, fact-checking organisations) say, then answer.

Return ONLY a JSON object (no markdown) with exactly these keys:
{
 "language": "ISO 639-1 code of the language the user wrote in, e.g. en, hy, ru",
 "type": "question | claim | headline | opinion | other",
 "verdict": "true | mostly_true | misleading | false | unproven | opinion | unanswerable",
 "confidence": 0-100,
 "answer": "1-2 sentence direct answer to the question or claim, in the user's language",
 "explanation": "3-6 sentences with the evidence, in the user's language. Separate what is proven from what is only speculation.",
 "claims": [{"claim": "...", "verdict": "true | mostly_true | misleading | false | unverifiable", "evidence": "1-2 sentences"}],
 "missing_context": "caveats, nuance or common misconceptions, or an empty string"
}
Rules:
- Base the answer on what the search results say, not on guesses. Never invent sources.
- "unproven" = no reliable evidence either way (for example an unconfirmed extraordinary claim). Use "false" only when reliable sources contradict it. Use "opinion" for matters of taste or values.
- Keep the JSON keys and verdict words in English. Write everything else in the user's language.
- At most 5 claims.

TEXT:
\"\"\"
__TEXT__
\"\"\""""


def _bot_claim_check(self, text):
    if not self._ensure_client():
        raise RuntimeError(self.last_error or "Gemini is not available.")
    prompt = (CLAIM_PROMPT.replace("__TODAY__", _dt.date.today().isoformat())
              .replace("__TEXT__", text))
    models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
    last, discovered = None, False
    while True:
        for m in models:
            for with_search in (True, False):
                try:
                    cfg = genai_types.GenerateContentConfig(
                        temperature=0.1, max_output_tokens=6000,
                        tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())] if with_search else None)
                    resp = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                    txt = (getattr(resp, "text", None) or "").strip()
                    if not txt:
                        last = f"{m}: empty answer"
                        continue
                    data = _parse_json_loose(txt)
                    self.chat_model = m
                    return data, _grounding_sources(resp), with_search
                except Exception as e:
                    last = f"{m}: {e}"
                    self.last_error = last
        if discovered:
            break
        discovered = True
        found, _ = self._discover_models()
        models = [x for x in found if x not in models][:4]
        if not models:
            break
    raise RuntimeError(last or "Gemini could not check this.")


SimpleChatbot.claim_check = _bot_claim_check

CLAIM_PTS2 = {"true": 95, "mostly_true": 78, "unproven": 45, "opinion": 50,
              "unanswerable": 45, "misleading": 25, "false": 5}
CLAIM_LABELS = {"true": "TRUE", "mostly_true": "MOSTLY TRUE", "misleading": "MISLEADING", "false": "FALSE",
                "unproven": "UNPROVEN (NO RELIABLE EVIDENCE)", "opinion": "OPINION, NOT A FACT",
                "unanswerable": "COULD NOT BE VERIFIED"}


def _claim_recommend(verdict, searched):
    msg = {
        "true": "✓ Reliable sources support this. Open the evidence links and read them yourself.",
        "mostly_true": "👍 Mostly right, with some details that are off or uncertain. Read the missing context.",
        "misleading": "⚠️ Partly true but misleading. Read the missing context before you share it.",
        "false": "🚨 Reliable sources contradict this. Do not share it as fact.",
        "unproven": "❔ There is no reliable evidence either way. Treat it as unproven, not as proven or disproven.",
        "opinion": "💭 This is a matter of opinion or values, so it cannot be proven true or false.",
        "unanswerable": "❔ I could not find reliable sources to settle this. Try rephrasing it, or check Reuters, AP or BBC.",
    }.get(verdict, "")
    if not searched:
        msg = "⚠️ The web search did not run, so this answer comes from the AI's memory only. " + msg
    return msg


def _claim_check(self, text):
    bot = getattr(self, "chatbot", None)
    if bot is None:
        raise ValueError("The chatbot is not connected, so I cannot fact-check this.")
    data, web, searched = bot.claim_check(text[:1500])
    verdict = _fn_norm(data.get("verdict"))
    if verdict not in CLAIM_PTS2:
        verdict = "unanswerable"
    try:
        conf = max(0, min(100, int(float(data.get("confidence", 60)))))
    except Exception:
        conf = 60
    score = 50 + (CLAIM_PTS2[verdict] - 50) * (0.4 + 0.6 * conf / 100)
    if not searched:
        score = 50 + (score - 50) * 0.6
    score = int(round(max(0, min(100, score))))

    claims = []
    for c in (data.get("claims") or [])[:6]:
        if isinstance(c, dict):
            v = _fn_norm(c.get("verdict"))
            claims.append({"claim": str(c.get("claim", ""))[:300],
                           "verdict": v if (v in CLAIM_PTS or v == "unverifiable") else "unverifiable",
                           "evidence": str(c.get("evidence", ""))[:500]})
    notes = []
    if not searched:
        notes.append("Google Search was not available, so this was answered from the AI's memory only. Trust it less.")
    answer = str(data.get("answer") or "").strip()
    expl = str(data.get("explanation") or "").strip()
    level = "High" if (searched and conf >= 70) else ("Medium" if searched else "Low")
    return {
        "success": True, "mode": "claim",
        "sources_found": [], "source_analysis": {}, "suspicious_patterns": [],
        "sentiment_analysis": None,
        "credibility_score": score,
        "verdict": CLAIM_LABELS[verdict],
        "confidence_level": level,
        "is_satire": False,
        "recommendation": _claim_recommend(verdict, searched),
        "fact_check": {
            "summary": (answer + " " + expl).strip(),
            "article_type": str(data.get("type", "")),
            "overall": verdict,
            "claims": claims,
            "manipulation": [],
            "missing_context": str(data.get("missing_context", "")),
        },
        "web_sources": web[:8],
        "signals": {"fact_check": score, "style": None, "ai_model": None},
        "fetched_from": None,
        "detected_language": str(data.get("language") or "").strip().lower().split("-")[0],
        "notes": notes,
    }


_fn_prev_detect2 = FakeNewsDetector.detect


def _fn_detect2(self, text):
    text = (text or "").strip()
    mode = (getattr(_LANG_TL, "news_mode", None) or "auto").lower()
    is_url = bool(re.fullmatch(r"<?(https?://[^\s<>]+)>?", text))
    claim = (not is_url) and (mode == "claim" or
                              (mode == "auto" and (len(text) < 350 or (text.endswith("?") and len(text) < 700))))
    if claim:
        res = _claim_check(self, text)
    else:
        res = _fn_prev_detect2(self, text)
        res["mode"] = "article"
    bot = getattr(self, "chatbot", None)
    if bot is not None and _lang_code() == "auto" and not res.get("detected_language"):
        sample = ((res.get("fact_check") or {}).get("summary")) or text
        res["detected_language"] = _detect_language(bot, sample)
    return res


FakeNewsDetector.detect = _fn_detect2

# ---- the chatbot also verifies "is it true / is it real" questions ----
_lang2_prev_bsp = SimpleChatbot._build_system_prompt


def _bsp_fact(self, message):
    prompt, qvec = _lang2_prev_bsp(self, message)
    prompt += ("\n\nFACT-CHECKING: When the user asks whether something is true, real, a myth or a rumor "
               "(for example 'are aliens real?'), do not answer from memory only. Use Google Search, start with a "
               "clear verdict (true / mostly true / misleading / false / unproven), then give a short evidence-based "
               "explanation, and separate what is proven from what is only speculation.")
    return prompt, qvec


SimpleChatbot._build_system_prompt = _bsp_fact

# ---------------------------------------------------------------- page: searchable language lists
LANG2_CSS = r"""
.lang-combo{position:relative;display:inline-block}
.lang-global{position:fixed;top:20px;right:122px;z-index:60}
@media (max-width:700px){.lang-global{top:calc(10px + var(--safe-t,0px));right:96px}}
.lang-btn{padding:7px 12px;border-radius:12px;cursor:pointer;font-size:14px;max-width:170px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;color:#e0e0e0;background:rgba(0,0,0,.55);border:2px solid rgba(221,132,72,.5)}
.lang-btn:hover{border-color:#ffaa5c;color:#fff}
.lang-panel{position:absolute;top:calc(100% + 6px);left:0;z-index:80;width:270px;max-width:86vw;padding:8px;border-radius:14px;
  background:rgba(10,12,22,.97);border:2px solid rgba(221,132,72,.6);box-shadow:0 10px 30px rgba(0,0,0,.6);white-space:normal}
.lang-panel[hidden]{display:none}
.lang-panel .lang-search{width:100%;padding:9px 12px;font-size:16px;border-radius:10px;margin:0}
.lang-list{max-height:260px;overflow-y:auto;margin-top:6px;scrollbar-width:thin}
.lang-item{padding:8px 10px;border-radius:8px;cursor:pointer;display:flex;justify-content:space-between;gap:10px;color:#e0e0e0;font-size:15px}
.lang-item small{opacity:.7;font-size:.9em;text-align:right}
.lang-item.active{background:rgba(221,132,72,.35)}
.lang-item.sel{font-weight:700}
.lang-empty{padding:10px;opacity:.7;color:#e0e0e0}
body.light .lang-btn{color:#1f4e8c;background:rgba(255,255,255,.7);border-color:rgba(60,120,200,.5)}
body.light .lang-panel{background:rgba(255,255,255,.97);border-color:rgba(60,120,200,.5)}
body.light .lang-item,body.light .lang-empty{color:#12304f}
body.light .lang-item.active{background:rgba(255,122,162,.3)}
"""

LANG2_JS = r"""
(function () {
  const $ = id => document.getElementById(id);
  const norm = s => String(s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  const native = c => {
    try { if (c === 'auto') return ''; const n = new Intl.DisplayNames([c], { type: 'language' }).of(c); return n && n !== c ? n : ''; }
    catch (e) { return ''; }
  };

  // the browser language is the fallback for tabs that have no text to detect from (Auto mode)
  try {
    document.cookie = 'pipbrowser=' + encodeURIComponent((navigator.language || 'en').split('-')[0].toLowerCase()) +
                      '; path=/; max-age=31536000; SameSite=Lax';
  } catch (e) {}

  // ---------- turn a <select> into a search box + list ----------
  let openCombo = null;
  document.addEventListener('mousedown', e => { if (openCombo && !openCombo.wrap.contains(e.target)) openCombo.close(); });

  function makeSearchable(select, isGlobal) {
    if (!select || select.__combo) return;
    const wrap = document.createElement('div');
    wrap.className = 'lang-combo' + (isGlobal ? ' lang-global' : '') + (select.classList.contains('vopt') ? ' vopt' : '');
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'lang-btn'; btn.title = select.title || 'Language';
    const panel = document.createElement('div'); panel.className = 'lang-panel'; panel.hidden = true;
    const search = document.createElement('input');
    search.type = 'text'; search.className = 'lang-search'; search.placeholder = '🔍 Search language…';
    search.autocomplete = 'off'; search.spellcheck = false;
    const list = document.createElement('div'); list.className = 'lang-list';
    panel.appendChild(search); panel.appendChild(list);
    wrap.appendChild(btn); wrap.appendChild(panel);
    select.parentNode.insertBefore(wrap, select);
    select.style.display = 'none';
    select.__combo = true;

    let items = [], shown = [], active = 0;
    const labelOf = o => o.value === 'auto' ? '🌐 Auto-detect' : o.text;
    function loadItems() {
      items = Array.from(select.options).map(o => ({
        value: o.value, name: labelOf(o), nat: native(o.value),
        key: norm(o.text + ' ' + native(o.value) + ' ' + o.value)
      }));
    }
    function refreshBtn() {
      const o = select.options[select.selectedIndex];
      const t = (o ? labelOf(o) : 'Language') + ' ▾';
      if (btn.textContent !== t) btn.textContent = t;
    }
    function render() {
      const q = norm(search.value.trim());
      shown = q ? items.filter(i => i.key.includes(q)) : items.slice();
      if (q) shown.sort((a, b) => (norm(b.name).startsWith(q) ? 1 : 0) - (norm(a.name).startsWith(q) ? 1 : 0));
      if (active >= shown.length) active = 0;
      list.innerHTML = '';
      if (!shown.length) { list.innerHTML = '<div class="lang-empty">No language found</div>'; return; }
      shown.forEach((i, idx) => {
        const d = document.createElement('div');
        d.className = 'lang-item' + (idx === active ? ' active' : '') + (i.value === select.value ? ' sel' : '');
        d.dataset.value = i.value;
        const a = document.createElement('span'); a.textContent = (i.value === select.value ? '✓ ' : '') + i.name;
        d.appendChild(a);
        if (i.nat && norm(i.nat) !== norm(i.name)) { const s = document.createElement('small'); s.textContent = i.nat; d.appendChild(s); }
        list.appendChild(d);
      });
      const el = list.querySelector('.active');
      if (el) el.scrollIntoView({ block: 'nearest' });
    }
    function choose(v) {
      select.value = v;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      refreshBtn(); api.close();
    }
    const api = {
      wrap,
      close() { panel.hidden = true; if (openCombo === api) openCombo = null; },
      open() {
        if (openCombo && openCombo !== api) openCombo.close();
        loadItems(); search.value = ''; active = Math.max(0, items.findIndex(i => i.value === select.value));
        const r = wrap.getBoundingClientRect();
        if (r.left + 280 > window.innerWidth) { panel.style.left = 'auto'; panel.style.right = '0'; }
        else { panel.style.left = '0'; panel.style.right = 'auto'; }
        panel.hidden = false; openCombo = api; render(); search.focus();
      }
    };
    btn.addEventListener('click', () => { panel.hidden ? api.open() : api.close(); });
    search.addEventListener('input', () => { active = 0; render(); });
    search.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(shown.length - 1, active + 1); render(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(0, active - 1); render(); }
      else if (e.key === 'Enter') { e.preventDefault(); if (shown[active]) choose(shown[active].value); }
      else if (e.key === 'Escape') { e.stopPropagation(); api.close(); btn.focus(); }
    });
    list.addEventListener('mousedown', e => {
      const it = e.target.closest('.lang-item');
      if (it) { e.preventDefault(); choose(it.dataset.value); }
    });
    new MutationObserver(() => { loadItems(); refreshBtn(); if (!panel.hidden) render(); }).observe(select, { childList: true });
    select.addEventListener('change', refreshBtn);
    setInterval(refreshBtn, 500);          // follows changes made by code (the two pickers stay in sync)
    loadItems(); refreshBtn();
  }

  makeSearchable($('pip-lang'), true);
  makeSearchable($('voice-lang'), false);
  window.__pipLangCombo = makeSearchable;   // lets later patches give other pickers the same search box

  // ---------- Fake News: article, link, or just a question ----------
  const ni = $('news-input');
  if (ni) {
    ni.placeholder = 'Paste an article or a link, or just ask: "Are aliens real?" / "Is it true that…?"';
    const grp = ni.closest('.input-group');
    const row = document.createElement('div');
    row.className = 'input-group';
    row.innerHTML = '<label>What to check</label><select id="news-mode">' +
      '<option value="auto">Auto-detect (article, link, or question)</option>' +
      '<option value="article">A full article</option>' +
      '<option value="claim">A question or claim (fact-check it on the web)</option></select>';
    grp.parentNode.insertBefore(row, grp);

    const base = window.fetch.bind(window);
    window.fetch = function (url, opts) {
      if (url === '/api/detect_fake_news' && opts && typeof opts.body === 'string') {
        try {
          const b = JSON.parse(opts.body);
          b.mode = $('news-mode').value;
          opts = Object.assign({}, opts, { body: JSON.stringify(b) });
        } catch (e) {}
      }
      return base(url, opts);
    };
  }
})();
"""

_lang2_prev_get_html = WebUI.get_html


def _lang2_get_html(self) -> str:
    html = _lang2_prev_get_html(self)
    html = html.replace('</head>', '<style>' + LANG2_CSS + '</style></head>', 1)
    return html.replace('</body>', '<script>' + LANG2_JS + '</script></body>', 1)


WebUI.get_html = _lang2_get_html

# ============================================================================
# IMAGE TOOLS PATCH: OCR (text -> TXT/Word/PDF), medical scan analysis for
# any body image, and a chat + search panel for the uploaded image.
# Paste ONCE, above the MAIN EXECUTION block.
# ============================================================================
from PIL import ImageOps

# Part 2 needs these two names. If the language block is missing, define simple versions.
if "_LANG_TL" not in globals():
    _LANG_TL = threading.local()
if "_lang_code" not in globals():
    def _lang_code():
        return "auto"

# ---- extra image types so "auto" can route documents and other scans ----
ImageAnalyzer.TYPE_LABELS = {
    "an MRI scan of a human brain": "brain_mri",
    "an MRI scan of a knee, spine or other body part": "other_mri",
    "a CT scan": "ct",
    "an X-ray image": "xray",
    "an ultrasound image": "ultrasound",
    "a mammogram": "mammogram",
    "a close-up photo of a skin lesion or mole": "skin",
    "a microscope slide of tissue cells": "pathology",
    "a photo or scan of a page, receipt, sign or screenshot full of text": "document",
    "a normal color photograph": "photo",
    "a drawing or illustration": "drawing",
}
_MEDICAL_TYPES = {"other_mri", "ct", "xray", "ultrasound", "mammogram", "skin", "pathology"}


def _parse_json_lenient(txt):
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return _json.loads(t, strict=False)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return _json.loads(m.group(0), strict=False)
        raise


def _load_hi(self, data, filename, spacing):
    """Loads the image at higher resolution (up to 2400 px) so small text stays readable."""
    notes = []
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        img, sp, note = self._load_image(data, filename, spacing)
        if note:
            notes.append(note)
        return img, sp, notes
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    scale = max(im.size) / 2400 if max(im.size) > 2400 else 1.0
    if scale > 1:
        im = im.resize((int(im.width / scale), int(im.height / scale)), Image.LANCZOS)
    return im, (spacing * scale if spacing else None), notes


# ---------------------------------------------------------------- OCR
OCR_PROMPT = """Read ALL the text visible in this image (printed, handwritten, signs, labels, tables, screenshots), in ANY language or script.
Rules:
- Copy the text exactly as written. Do not translate, correct or summarize it.
- Keep the natural reading order, line breaks and paragraphs. Show tables with the columns separated by a TAB character.
- If a part is unreadable, write [unreadable] there.
- Do not add anything that is not in the image.
Return ONLY a JSON object: {"has_text": true, "language": "name of the main language", "text": "the full text", "handwritten": false, "notes": "short note or empty string"}"""


def _ocr_extract(self, img, P):
    P("Reading the text in the image", 40)
    bot = getattr(self, "chatbot", None)
    text, lang, hw, note, source, err = "", "", False, "", "", None
    if bot is not None:
        try:
            data = _parse_json_lenient(bot.vision_json(_to_jpeg(img, 95), OCR_PROMPT))
            text = str(data.get("text") or "").strip()
            lang = str(data.get("language") or "")
            hw = bool(data.get("handwritten"))
            note = str(data.get("notes") or "")
            source = "Gemini vision"
        except Exception as e:
            err = e
            print("Gemini OCR failed:", str(e)[:200])
    if not text:
        try:
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
            source = "Tesseract OCR (backup)"
        except Exception as e2:
            if err is None:
                err = e2
    P("Preparing the text", 90)
    if not text and err is not None:
        return {"success": False, "error": "Could not read the text: " + str(err)[:200]}
    return {"success": True, "kind": "ocr", "has_text": bool(text), "text": text, "language": lang,
            "handwritten": hw, "ocr_note": note, "source": source or "none",
            "word_count": len(text.split()), "char_count": len(text)}


# ---------------------------------------------------------------- medical scan
MEDICAL_PROMPT = """You are an AI assistant that helps screen a medical or body image for possible tumors, masses and other abnormalities. This is a screening aid for a clinician to review, NOT a diagnosis.
Look carefully at the image. It can be any kind: MRI, CT, X-ray, mammogram, ultrasound, PET, endoscopy, skin photo or dermoscopy, microscope or pathology slide, or a photo of a body part.
Return ONLY a JSON object with these keys:
{
 "is_medical_image": true,
 "modality": "e.g. chest X-ray, brain MRI, CT abdomen, skin photo",
 "body_part": "e.g. lung, brain, breast, skin",
 "image_quality": "good, fair or poor, with one short reason",
 "tumor_status": "yes | no | uncertain",
 "impression": "2-4 plain sentences: what you see and what it could mean, using careful words like 'may' or 'could'",
 "findings": [
  {"label": "short name", "type": "tumor | mass | nodule | lesion | cyst | fracture | inflammation | other | normal",
   "suspicion": "low | moderate | high", "location": "where in the image / organ",
   "size_estimate": "only if a scale is visible, otherwise 'cannot be measured without a scale'",
   "description": "what it looks like, and the possible tumor type if you can reasonably say (for example glioma, meningioma, adenocarcinoma, melanoma) with careful wording",
   "box_2d": [ymin, xmin, ymax, xmax]}
 ],
 "recommendations": ["practical next steps, for example show it to a doctor, a follow-up scan, a biopsy opinion"]
}
Rules:
- box_2d uses coordinates normalized to 0-1000 (top-left is 0,0). One finding per abnormal area.
- Report only what you can really see. Never invent findings. If the image looks normal, use tumor_status "no" and an empty findings list. If you cannot tell, use "uncertain".
- If this is not a medical image, set is_medical_image to false, tumor_status to "uncertain" and explain in impression.
- Never state that someone definitely has cancer."""

_SUSP_COLOR = {"high": (220, 38, 38), "moderate": (249, 115, 22), "low": (234, 179, 8)}


def _skin_specialist(self, img):
    name = os.getenv("SKIN_MODEL", "Anwarkh1/Skin_Cancer-Image_Classification")
    if not name or name.lower() == "off":
        return None
    try:
        pipe = self._pipes.get("skin")
        if pipe is None:
            kw = {"torch_dtype": torch.float16} if self.device == 0 else {}
            pipe = pipeline("image-classification", model=name, device=self.device, **kw)
            self._pipes["skin"] = pipe
        out = pipe(img, top_k=4)
        return {"model": name, "predictions": [{"label": str(o["label"]), "score": round(float(o["score"]) * 100, 1)} for o in out]}
    except Exception as e:
        print("Skin model not available:", str(e)[:150])
        return None


def _medical_analyze(self, img, spacing, P):
    bot = getattr(self, "chatbot", None)
    if bot is None:
        raise RuntimeError("Gemini (the chatbot) is not connected, so the scan cannot be analysed.")
    P("Asking the AI to examine the scan", 35)
    data = _parse_json_lenient(bot.vision_json(_to_jpeg(img, 92), MEDICAL_PROMPT))
    W, H = img.size
    P("Marking the findings", 80)
    out_img = img.copy()
    dr = ImageDraw.Draw(out_img)
    font = self._font(max(14, int(min(W, H) / 34)))
    lw = max(3, int(min(W, H) / 190))
    findings, n = [], 0
    for f in (data.get("findings") or []):
        if not isinstance(f, dict):
            continue
        n += 1
        susp = str(f.get("suspicion") or "low").lower()
        if susp not in _SUSP_COLOR:
            susp = "low"
        e = {"number": n, "label": str(f.get("label") or "finding"), "type": str(f.get("type") or "other").lower(),
             "suspicion": susp, "location": str(f.get("location") or ""),
             "size_estimate": str(f.get("size_estimate") or ""), "description": str(f.get("description") or ""),
             "width_px": None, "height_px": None, "position": None, "size_mm": None}
        box = _norm_box(f.get("box_2d"), W, H)
        if box:
            e.update(_box_meta(self, box, W, H))
            x0, y0, x1, y1 = box
            if spacing:
                e["size_mm"] = f"{(x1 - x0) * spacing:.1f} x {(y1 - y0) * spacing:.1f} mm"
            col = _SUSP_COLOR[susp]
            dr.rectangle([x0, y0, x1, y1], outline=col, width=lw)
            text = f"{n}. {e['label']}"
            tb = dr.textbbox((0, 0), text, font=font)
            th = tb[3] - tb[1] + 8
            ty = y0 - th if y0 - th > 0 else y0
            dr.rectangle([x0, ty, x0 + tb[2] - tb[0] + 10, ty + th], fill=col)
            dr.text((x0 + 5, ty + 3), text, fill=(255, 255, 255), font=font)
        findings.append(e)
    status = str(data.get("tumor_status") or "uncertain").lower()
    if status not in ("yes", "no", "uncertain"):
        status = "uncertain"
    modality = str(data.get("modality") or "")
    body = str(data.get("body_part") or "")
    res = {"success": True, "kind": "medical", "source": "Gemini vision (AI screening aid)",
           "is_medical_image": data.get("is_medical_image") is not False,
           "modality": modality, "body_part": body, "image_quality": str(data.get("image_quality") or ""),
           "tumor_status": status, "impression": str(data.get("impression") or ""),
           "findings": findings, "recommendations": [str(r) for r in (data.get("recommendations") or [])],
           "annotated_image": self._to_data_url(out_img),
           "notes": [] if spacing else ["Sizes are only given in millimetres when you enter the pixel spacing (mm/pixel) or upload a DICOM file."],
           "disclaimer": "This is an AI screening aid, not a medical diagnosis. A radiologist or doctor must review the images."}
    mod = (modality + " " + body).lower()
    if "brain" in mod and "mri" in mod:
        res["_brain_mri"] = True
    if any(k in mod for k in ("skin", "dermat", "mole", "melanoma")):
        P("Checking with the skin-lesion model", 90)
        sp = _skin_specialist(self, img)
        if sp:
            res["specialist"] = sp
    return res


# ---------------------------------------------------------------- AI-generated image check
AIG_MODEL = os.getenv("AIG_MODEL", "umm-maybe/AI-image-detector")
AIG_PROMPT = """You are an image-forensics expert. Look carefully at this image and judge whether it was made by a camera (real photo) or generated by AI (Midjourney, DALL-E, Stable Diffusion, GANs, deepfakes etc.).
Check visual tells: impossible or melted anatomy/hands, inconsistent lighting or shadows, weird or garbled text, overly smooth or plastic skin, repeating textures, warped backgrounds, noise/grain patterns that look painted, watermarks, and metadata-like artifacts.
Be honest about uncertainty - many images are genuinely hard to call.
Return ONLY a JSON object (no markdown):
{"verdict": "real | ai | uncertain",
 "confidence": 0-100,
 "summary": "2-4 sentences explaining your judgement in plain words",
 "signs": [{"sign": "the visual clue you noticed", "points_to": "ai | real | neutral"}]}"""


def _aig_score_hf(self, img):
    """(ai_pct, real_pct, model_name) from a Hugging Face detector, or (None, None, note)."""
    try:
        name = AIG_MODEL
        clf = self._pipe("aig", "image-classification", name)
        out = clf(img)
        ai_s = real_s = 0.0
        ai_re = re.compile(r"\b(ai|fake|generated|synthetic|artificial|gan|diffusion|deepfake|sdxl|midjourney|dalle|fake image)\b")
        real_re = re.compile(r"\b(real|human|natural|photo|photograph|camera|authentic|genuine|real image)\b")
        for d in out:
            lab = str(d.get("label", "")).lower()
            sc = float(d.get("score", 0) or 0)
            if ai_re.search(lab):
                ai_s += sc
            elif real_re.search(lab):
                real_s += sc
        if ai_s + real_s <= 0:
            raise ValueError("unrecognised labels: " + ",".join(str(d.get("label")) for d in out[:5]))
        s = ai_s + real_s
        return round(ai_s / s * 100, 1), round(real_s / s * 100, 1), name
    except Exception as e:
        print("AIG model failed:", str(e)[:150])
    try:
        clip = self._pipe("clip", "zero-shot-image-classification", self.CLIP_MODEL)
        out = clip(img, candidate_labels=[
            "a real photograph taken by a camera",
            "an AI generated or synthetic image",
            "a digital artwork, painting or illustration",
            "a screenshot, graphic or cartoon"])
        scores = {str(d["label"]): float(d["score"]) for d in out}
        ai_p = (scores.get("an AI generated or synthetic image", 0.0)) * 100
        real_p = scores.get("a real photograph taken by a camera", 0.0) * 100
        return round(ai_p, 1), round(real_p, 1), "CLIP zero-shot (" + self.CLIP_MODEL + ")"
    except Exception as e:
        return None, None, "no HF detector: " + str(e)[:100]


def _aig_detect(self, img, P):
    """AI-generated vs real photo check: HF model score + vision-model explanation."""
    res = {"success": True, "kind": "aidetect", "score": None, "real_score": None,
           "verdict": "uncertain", "confidence": 0, "summary": "", "reasons": [],
           "signs": [], "model": "", "notes": []}
    P("Checking with the AI-image detector", 25)
    ai_p, real_p, model = _aig_score_hf(self, img)
    res["model"] = model
    if ai_p is None:
        res["notes"].append("No Hugging Face detector could run, so only the visual analysis below is available.")
    else:
        res["score"], res["real_score"] = ai_p, real_p
        if "CLIP" in model:
            res["notes"].append("This used a CLIP zero-shot estimate, not a dedicated AI-image model. Treat the percentage as a rough guide.")

    P("Asking the vision model for the visual tells", 60)
    bot = getattr(self, "chatbot", None)
    vision = None
    if bot is not None:
        try:
            vision = _parse_json_lenient(bot.vision_json(_to_jpeg(img, 90), AIG_PROMPT))
        except Exception as e:
            res["notes"].append("The vision explanation was unavailable: " + str(e)[:100])
    if isinstance(vision, dict):
        res["summary"] = str(vision.get("summary") or "").strip()
        try:
            res["confidence"] = max(0, min(100, int(vision.get("confidence") or 0)))
        except Exception:
            pass
        vv = str(vision.get("verdict") or "").lower()
        if vv in ("real", "ai", "uncertain"):
            res["verdict"] = vv
        signs = vision.get("signs") or []
        res["signs"] = [{"sign": str(s.get("sign") or ""), "points_to": str(s.get("points_to") or "neutral")}
                        for s in signs if isinstance(s, dict)][:12]
        for s in res["signs"]:
            pt = s["points_to"]
            if pt in ("ai", "real") and s["sign"]:
                res["reasons"].append(("🤖 " if pt == "ai" else "📷 ") + s["sign"] +
                                      (" — points to AI generation" if pt == "ai" else " — points to a real photo"))
    if ai_p is not None and res["verdict"] == "uncertain" and not res["confidence"]:
        res["confidence"] = round(max(ai_p, real_p or 0), 1)
    if res["score"] is not None and res["verdict"] == "uncertain" and not vision:
        res["verdict"] = "ai" if res["score"] >= 65 else "real" if res["score"] <= 35 else "uncertain"
    if res["verdict"] == "uncertain" and res["score"] is not None and not res["summary"]:
        res["summary"] = (f"The detector gives this image a {res['score']}% AI-likelihood score "
                          f"({res['real_score']}% real). That is in the uncertain range - many images are genuinely hard to call.")
    return res


# ---------------------------------------------------------------- analyze() router
_more_prev_analyze = ImageAnalyzer.analyze


def _share_more(self, data, filename, res):
    bot = getattr(self, "chatbot", None)
    if not bot or not res.get("success"):
        return
    try:
        if res.get("kind") == "ocr":
            s = "Text read from the image: " + (res.get("text") or "")[:6000]
        else:
            s = "Medical screening of " + (res.get("modality") or "an image") + ": " + (res.get("impression") or "")
            for f in res.get("findings", [])[:10]:
                s += f" Finding {f['number']}: {f['label']} ({f['type']}, {f['suspicion']} suspicion) {f['location']}."
        bot.set_image_context(_jpeg_from_upload(data, filename), s)
    except Exception as e:
        print("Could not share with chatbot:", e)


def _analyze_more(self, data, filename="", mode="auto", spacing=None, progress=None):
    def P(s, p):
        if progress:
            progress(s, p)

    if mode == "aidetect":
        try:
            with self._job_lock:
                self._cb = progress
                try:
                    img, sp, notes = _load_hi(self, data, filename, spacing)
                    res = _aig_detect(self, img, P)
                finally:
                    self._cb = None
        except Exception as e:
            return {"success": False, "error": str(e)}
        if res.get("success") and notes:
            res["notes"] = notes + res.get("notes", [])
        return res

    if mode not in ("auto", "ocr", "medical"):
        return _more_prev_analyze(self, data, filename, mode, spacing, progress)

    itype = None
    if mode == "auto":
        if filename and filename.lower().endswith((".dcm", ".dicom")):
            return _more_prev_analyze(self, data, filename, "auto", spacing, progress)
        try:
            P("Identifying the image type", 12)
            with self._job_lock:
                self._cb = progress
                try:
                    im0 = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                    im0.thumbnail((1280, 1280))
                    itype = self._image_type(im0)
                finally:
                    self._cb = None
        except Exception as e:
            print("Type detection failed:", str(e)[:150])
            return _more_prev_analyze(self, data, filename, "auto", spacing, progress)
        t = itype["type"]
        if t == "brain_mri":
            res = _more_prev_analyze(self, data, filename, "mri", spacing, progress)
        elif t in _MEDICAL_TYPES:
            mode = "medical"
            res = None
        elif t == "document":
            mode = "ocr"
            res = None
        else:
            res = _more_prev_analyze(self, data, filename, "objects", spacing, progress)
        if res is not None:
            res.setdefault("image_type", itype)
            return res

    try:
        with self._job_lock:
            self._cb = progress
            try:
                img, sp, notes = _load_hi(self, data, filename, spacing)
                if mode == "ocr":
                    res = _ocr_extract(self, img, P)
                else:
                    res = _medical_analyze(self, img, sp, P)
            finally:
                self._cb = None
    except Exception as e:
        return {"success": False, "error": str(e)}

    if res.pop("_brain_mri", False):
        r2 = _more_prev_analyze(self, data, filename, "mri", spacing, progress)
        if r2.get("success"):
            msg = "Brain MRI detected, so the specialised brain-tumor models were used."
            r2.setdefault("notes", [])
            if msg not in r2["notes"]:
                r2["notes"].append(msg)
            return r2
    if res.get("success"):
        if itype:
            res["image_type"] = itype
        if notes and res.get("kind") == "medical":
            res["notes"] = notes + res.get("notes", [])
        _share_more(self, data, filename, res)
    return res


ImageAnalyzer.analyze = _analyze_more


# ---------------------------------------------------------------- export (TXT / Word / PDF)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _make_docx(text, title):
    from docx import Document
    d = Document()
    d.add_heading(_CTRL.sub("", title), 1)
    for line in _CTRL.sub("", text).split("\n"):
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _make_pdf(text, title):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = "Helvetica"
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("PipFont", p))
                font = "PipFont"
                break
            except Exception:
                pass
    body = ParagraphStyle("b", fontName=font, fontSize=11, leading=15)
    head = ParagraphStyle("t", fontName=font, fontSize=16, leading=20, spaceAfter=10)

    def clean(s):
        return _CTRL.sub("", s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story = [Paragraph(clean(title), head)]
    for line in text.split("\n"):
        story.append(Paragraph(clean(line).replace("\t", "&nbsp;" * 4) or "&nbsp;", body))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48).build(story)
    return buf.getvalue()


# ---------------------------------------------------------------- chat + search panel (server)
IMG_STORE = {}


def _store_put(img, jpeg):
    iid = uuid.uuid4().hex[:16]
    if len(IMG_STORE) >= 12:
        IMG_STORE.pop(next(iter(IMG_STORE)))
    IMG_STORE[iid] = {"img": img, "jpeg": jpeg, "hist": [], "t": time.time()}
    return iid


ASK_PROMPT = """You are a helpful visual assistant. You can see the image the user uploaded. The user's language is __LANG__. Always write your reply in __LANG__, even if the image contains text in other languages. If __LANG__ is English, reply in English unless the user clearly asked for a translation.
Earlier conversation about this image:
__HIST__

User's new message: __MSG__

Choose ONE action and return ONLY a JSON object (no markdown):
{"action": "answer | find | zoom | ocr | medical | web | aidetect",
 "reply": "your reply to the user, short and clear",
 "marks": [{"label": "what it is", "box_2d": [ymin, xmin, ymax, xmax]}],
 "zoom_box": [ymin, xmin, ymax, xmax] or null}
Actions:
- aidetect: the user asks if the image is AI-generated, fake, synthetic, deepfake, computer-made or not real, or wants its authenticity checked. (reply: one short sentence, the check result is added automatically)
- find: the user wants something located, searched for, counted or highlighted in the picture. Put one entry in marks for EVERY match (box_2d normalized 0-1000) and say how many you found. If there are none, leave marks empty and say so.
- zoom: the user wants a closer look at one region. Give zoom_box.
- ocr: the user wants the text in the image read, copied or turned into a document. (reply: one short sentence, the text is added automatically)
- medical: the user wants a tumor, cancer, disease or abnormality scan of a medical image. (reply: one short sentence, the scan result is added automatically)
- web: the user asks for facts from the internet about what is in the image (prices, who made it, what a landmark is, news).
- answer: any other question about the image.
Rules: never identify real people from their faces. Only describe what you can really see. Do not invent."""


def _draw_marks(an, img, marks):
    out = img.copy().convert("RGB")
    W, H = out.size
    d = ImageDraw.Draw(out)
    font = an._font(max(14, int(min(W, H) / 34)))
    lw = max(3, int(min(W, H) / 200))
    n = 0
    for m in marks:
        if not isinstance(m, dict):
            continue
        box = _norm_box(m.get("box_2d"), W, H)
        if not box:
            continue
        n += 1
        label = str(m.get("label") or "match")
        col = an._color(label)
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        text = f"{n}. {label}"
        tb = d.textbbox((0, 0), text, font=font)
        th = tb[3] - tb[1] + 8
        ty = y0 - th if y0 - th > 0 else y0
        d.rectangle([x0, ty, x0 + tb[2] - tb[0] + 10, ty + th], fill=col)
        d.text((x0 + 5, ty + 3), text, fill=(255, 255, 255), font=font)
    return out, n


def _ask_image(ui, iid, message):
    ent = IMG_STORE.get(iid)
    if not ent:
        raise ValueError("The image expired on the server. Choose it again.")
    an, bot, img = ui.image_analyzer, ui.chatbot, ent["img"]
    W, H = img.size
    hist = "\n".join(ent["hist"][-8:]) or "(nothing yet)"
    lang = _detect_language(bot, message)
    lang_name = LANG_NAMES.get(lang, "English")
    prompt = (ASK_PROMPT.replace("__HIST__", hist).replace("__MSG__", message)
              .replace("__LANG__", lang_name))
    data = _parse_json_lenient(bot.vision_json(_to_jpeg(img, 92), prompt))
    action = str(data.get("action") or "answer").lower()
    reply = str(data.get("reply") or "").strip()
    out = {"success": True, "action": action, "reply": reply, "image": None,
           "text": None, "medical": None, "aidetect": None, "language": lang}

    if action == "find":
        im, n = _draw_marks(an, img, data.get("marks") or [])
        if n:
            out["image"] = an._to_data_url(im)
        elif not reply:
            out["reply"] = "I could not find that in the image."
    elif action == "zoom":
        box = _norm_box(data.get("zoom_box"), W, H)
        if box:
            x0, y0, x1, y1 = box
            px, py = (x1 - x0) * 0.05, (y1 - y0) * 0.05
            crop = img.crop((max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)))
            s = 1000 / max(crop.size)
            if s > 1:
                crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
            out["image"] = an._to_data_url(crop)
    elif action == "ocr":
        r = _ocr_extract(an, img, lambda s, p: None)
        if not r.get("success"):
            raise ValueError(r.get("error", "Could not read the text."))
        out["text"] = r["text"]
        out["reply"] = reply or ("Here is the text I found." if r["text"] else "I could not find any text.")
    elif action == "medical":
        r = _medical_analyze(an, img, None, lambda s, p: None)
        r.pop("_brain_mri", None)
        out["medical"] = r
        out["reply"] = reply or "Here is the scan analysis."
    elif action == "aidetect":
        r = _aig_detect(an, img, lambda s, p: None)
        out["aidetect"] = r
        out["reply"] = reply or (r.get("summary") or "Here is the AI-generation check.")
    elif action == "web":
        bot.set_image_context(ent["jpeg"], "The user is asking about this uploaded image.")
        lang_name = LANG_NAMES.get(lang, "")
        q = f"(Please answer in {lang_name}.) {message}" if lang_name else message
        out["reply"] = bot.chat(q)

    if not out["reply"]:
        out["reply"] = "Done."
    ent["hist"].append("User: " + message[:300])
    ent["hist"].append("Assistant: " + out["reply"][:300])
    return out


def _ask_free(ui, message):
    """Image-chat used without an image: plain chatbot answers, still multilingual."""
    bot = ui.chatbot
    lang = _detect_language(bot, message)
    lang_name = LANG_NAMES.get(lang, "English")
    q = f"(Please answer in {lang_name}.) {message}"
    reply = bot.chat(q)
    return {"success": True, "action": "chat", "reply": reply or "I could not answer right now.",
            "image": None, "text": None, "medical": None, "language": lang}


def _more_add_routes(self):
    from flask import Response

    @self.app.route('/api/image_upload', methods=['POST'])
    def image_upload():
        f = request.files.get('image')
        if not f:
            return jsonify({'success': False, 'error': 'No image received.'}), 400
        data, name = f.read(), f.filename or ""
        try:
            if name.lower().endswith((".dcm", ".dicom")):
                img, _, _ = self.image_analyzer._load_image(data, name, None)
            else:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
                img.thumbnail((2400, 2400), Image.LANCZOS)
        except Exception as e:
            return jsonify({'success': False, 'error': 'Could not read the image: ' + str(e)[:120]}), 400
        iid = _store_put(img, _to_jpeg(img, 90))
        return jsonify({'success': True, 'id': iid, 'width': img.width, 'height': img.height})

    @self.app.route('/api/image_ask_start', methods=['POST'])
    def image_ask_start():
        d = request.get_json(silent=True) or {}
        iid, msg = str(d.get("id") or ""), str(d.get("message") or "").strip()
        if not msg:
            return jsonify({'success': False, 'error': 'Please type a message.'}), 400
        if iid:
            if iid not in IMG_STORE:
                return jsonify({'success': False, 'error': 'The image expired on the server. Choose it again.'}), 400
            return _job_start(lambda: _ask_image(self, iid, msg))
        return _job_start(lambda: _ask_free(self, msg))

    @self.app.route('/api/export_text', methods=['POST'])
    def export_text():
        d = request.get_json(silent=True) or {}
        text, fmt = str(d.get("text") or ""), str(d.get("format") or "txt").lower()
        title = str(d.get("title") or "Extracted text")[:120]
        if not text.strip():
            return jsonify({'success': False, 'error': 'There is no text to export.'}), 400
        try:
            if fmt == "txt":
                body, mime = text.encode("utf-8"), "text/plain; charset=utf-8"
            elif fmt == "docx":
                body, mime = _make_docx(text, title), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif fmt == "pdf":
                body, mime = _make_pdf(text, title), "application/pdf"
            else:
                return jsonify({'success': False, 'error': 'Unknown format.'}), 400
        except ImportError as e:
            return jsonify({'success': False, 'error': 'Missing package: ' + str(e) + '. Run: !pip install python-docx reportlab'}), 500
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)[:200]}), 500
        return Response(body, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="pip-ai-text.{fmt}"'})


_more_prev_init = WebUI.__init__


def _more_init(self):
    _more_prev_init(self)
    if FLASK_AVAILABLE:
        _more_add_routes(self)


WebUI.__init__ = _more_init

# ---------------------------------------------------------------- page
MORE_PANEL = r"""
<div id="img-ask">
  <div class="cr-title">💬 Chat with the image agent — ask questions or give commands</div>
  <div class="ia-chips">
    <button type="button" class="ia-chip" data-q="Read all the text in this image">📝 Read the text</button>
    <button type="button" class="ia-chip" data-q="Scan this image for tumors or any abnormal areas and mark them">🩺 Scan for tumors</button>
    <button type="button" class="ia-chip" data-q="Find and mark every person">👥 Find people</button>
    <button type="button" class="ia-chip" data-q="Describe this image in detail">👁️ Describe</button>
    <button type="button" class="ia-chip" data-q="Translate any text you can see into English">🌐 Translate text</button>
    <button type="button" class="ia-chip" data-q="List everything you can see in this image">📋 List everything</button>
    <button type="button" class="ia-chip" data-q="Zoom into the most important part of this image">🔍 Zoom in</button>
    <button type="button" class="ia-chip" data-q="Is this image AI-generated or real? Check if it is fake">🤖 Real or AI?</button>
  </div>
  <div class="ia-box" id="ia-box"></div>
  <div class="composer">
    <textarea id="ia-input" rows="1" placeholder="Ask anything — about the image or anything else: &quot;find all red cars&quot;, &quot;where is the price?&quot;, &quot;what is the capital of Armenia?&quot;"></textarea>
    <button type="button" class="generate-button compact" id="ia-send"><span>Ask</span></button>
  </div>
  <div class="ia-hint">Works in any language — just write or speak in yours. With an image loaded I can answer questions, find and mark things, zoom in, read text, scan medical images and look things up on the web. Without an image I'm the regular chatbot. 🎤 voice in · 🔊 read replies aloud.</div>
</div>
"""

MORE_CSS = r"""
#img-ask{margin-top:22px;white-space:normal}
.ia-box{max-height:520px;overflow-y:auto;padding:6px 4px;scrollbar-width:thin;scrollbar-color:rgba(221,132,72,.6) transparent}
.ia-box img.result{max-width:100%;border-radius:12px;margin:8px 0;border:2px solid rgba(255,255,255,.3);display:block}
.ia-chips{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px}
.ia-chip,.oc-btn{padding:7px 14px;border-radius:999px;cursor:pointer;font-size:.9em;color:#ffaa5c;
  background:rgba(0,0,0,.55);border:1px solid rgba(221,132,72,.6)}
.ia-chip:hover,.oc-btn:hover{background:rgba(221,132,72,.3);color:#fff}
.oc-btn:disabled{opacity:.5}
.ia-hint{font-size:.85em;opacity:.75;margin-top:6px}
.ocr-box textarea{min-height:220px;width:100%;font-family:'Segoe UI',Arial,sans-serif;line-height:1.5;white-space:pre-wrap}
.oc-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
#img-output details{margin-top:12px}
#img-output summary{cursor:pointer;font-weight:700}
body.light .ia-chip,body.light .oc-btn{color:#b8325a;background:rgba(255,255,255,.7);border-color:#ff7aa2}
"""

MORE_JS = r"""
(function () {
  const $ = id => document.getElementById(id);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => escapeHtml(String(s == null ? '' : s));
  const title = t => '<div class="cr-title">' + t + '</div>';
  const list = a => '<ul>' + a.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul>';

  // ---------- new modes in the dropdown ----------
  const modeSel = $('img-mode');
  if (modeSel) {
    modeSel.insertAdjacentHTML('beforeend',
      '<option value="medical">🩺 Medical scan (tumors, any body part)</option>' +
      '<option value="ocr">📝 Read text (OCR)</option>' +
      '<option value="aidetect">🤖 AI-generated image? (real vs fake)</option>');
  }

  // ---------- editable text box with copy / TXT / Word / PDF ----------
  function box(text, ttl) {
    return '<div class="ocr-box" data-title="' + esc(ttl) + '"><textarea spellcheck="false">' + esc(text) + '</textarea>' +
      '<div class="oc-row">' +
      '<button type="button" class="oc-btn" onclick="pipCopy(this)">📋 Copy</button>' +
      '<button type="button" class="oc-btn" onclick="pipExport(this,\'txt\')">⬇️ TXT</button>' +
      '<button type="button" class="oc-btn" onclick="pipExport(this,\'docx\')">⬇️ Word</button>' +
      '<button type="button" class="oc-btn" onclick="pipExport(this,\'pdf\')">⬇️ PDF</button></div></div>';
  }
  window.pipCopy = function (btn) {
    const t = btn.closest('.ocr-box').querySelector('textarea').value;
    navigator.clipboard.writeText(t).then(() => {
      const o = btn.textContent; btn.textContent = 'Copied ✓'; setTimeout(() => { btn.textContent = o; }, 1500);
    });
  };
  window.pipExport = async function (btn, fmt) {
    const b = btn.closest('.ocr-box');
    const text = b.querySelector('textarea').value;
    const old = btn.textContent; btn.disabled = true; btn.textContent = '…';
    try {
      const r = await fetch('/api/export_text', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, format: fmt, title: b.dataset.title || 'Extracted text' }) });
      if (!r.ok) {
        let m = 'Export failed.';
        try { m = (await r.json()).error || m; } catch (e) {}
        throw new Error(m);
      }
      const blob = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob); a.download = 'pip-ai-text.' + fmt;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 3000);
    } catch (e) { alert(e.message); }
    finally { btn.disabled = false; btn.textContent = old; }
  };

  // ---------- result screens ----------
  function renderOcr(d) {
    let h = '<div class="img-src">📝 ' + esc(d.source || '') + '</div>';
    if (!d.has_text) return h + '<div class="cr-status warn">❔ I could not find any text in this image.</div>';
    h += '<div class="cr-status ok">✅ Text found · ' + d.word_count + ' words' + (d.language ? ' · ' + esc(d.language) : '') +
         (d.handwritten ? ' · handwriting' : '') + '</div>';
    h += '<p><small>You can edit the text before you download it.</small></p>' + box(d.text, 'Text from image');
    if (d.ocr_note) h += '<p><small>' + esc(d.ocr_note) + '</small></p>';
    return h;
  }

  const SUSP = { high: '🔴 high', moderate: '🟠 moderate', low: '🟡 low' };
  function medText(d) {
    const L = ['Medical image screening report (AI screening aid, not a diagnosis)', ''];
    if (d.modality) L.push('Image: ' + d.modality + (d.body_part ? ' (' + d.body_part + ')' : ''));
    if (d.image_quality) L.push('Quality: ' + d.image_quality);
    L.push('Tumor status: ' + d.tumor_status, '', 'Impression:', d.impression || '-', '');
    (d.findings || []).forEach(f => {
      L.push(f.number + '. ' + f.label + ' [' + f.type + ', suspicion ' + f.suspicion + ']');
      if (f.location) L.push('   Location: ' + f.location);
      if (f.size_mm || f.size_estimate) L.push('   Size: ' + (f.size_mm || f.size_estimate));
      if (f.description) L.push('   ' + f.description);
    });
    if ((d.recommendations || []).length) { L.push('', 'Suggested next steps:'); d.recommendations.forEach(r => L.push('- ' + r)); }
    L.push('', d.disclaimer || '');
    return L.join('\n');
  }
  function renderMedical(d) {
    const T = { yes: ['warn', '⚠️ Possible tumor or mass found'], no: ['ok', '✅ No tumor seen'],
                uncertain: ['warn', '❔ Unclear, a doctor should look at it'] };
    const st = T[d.tumor_status] || T.uncertain;
    let h = '<div class="img-src">🩺 ' + esc(d.source || '') + '</div>';
    h += '<div class="cr-status ' + st[0] + '">' + st[1] + '</div>';
    const chips = [d.modality, d.body_part, d.image_quality ? 'quality: ' + d.image_quality : ''].filter(Boolean);
    if (chips.length) h += '<div class="ppl-chips">' + chips.map(c => '<span class="ppl-chip">' + esc(c) + '</span>').join('') + '</div>';
    if (d.annotated_image) h += '<img class="result" src="' + d.annotated_image + '">';
    if (d.impression) h += title('🧾 Impression') + '<p>' + esc(d.impression) + '</p>';
    if (d.findings && d.findings.length) {
      h += title('📍 Findings') + '<table><tr><th>#</th><th>Finding</th><th>Type</th><th>Suspicion</th><th>Location</th><th>Size</th><th>Details</th></tr>' +
        d.findings.map(f => '<tr><td>' + f.number + '</td><td>' + esc(f.label) + '</td><td>' + esc(f.type) + '</td><td>' + (SUSP[f.suspicion] || esc(f.suspicion)) +
          '</td><td>' + esc(f.location || f.position || '—') + '</td><td>' + esc(f.size_mm || f.size_estimate || '—') + '</td><td>' + esc(f.description || '—') + '</td></tr>').join('') + '</table>';
    }
    if (d.specialist) {
      h += title('🔬 Skin-lesion model (' + esc(d.specialist.model) + ')') + '<table>' +
        d.specialist.predictions.map(p => '<tr><td>' + esc(p.label) + '</td><td>' + p.score + '%</td><td><div class="img-bar"><i style="width:' + p.score + '%"></i></div></td></tr>').join('') + '</table>';
    }
    if (d.recommendations && d.recommendations.length) h += title('✅ Suggested next steps') + list(d.recommendations);
    if (d.notes && d.notes.length) h += title('📝 Notes') + list(d.notes);
    h += '<details><summary>📄 Report text (edit and download)</summary>' + box(medText(d), 'Medical image screening report') + '</details>';
    if (d.disclaimer) h += '<div class="img-disc">⚕️ ' + esc(d.disclaimer) + '</div>';
    return h;
  }
  function renderAiDetect(d) {
    const T = { real: ['ok', '✅ Likely a real photo'], uncertain: ['warn', '❔ Uncertain — could be either'],
                ai: ['warn', '🔴 Likely AI-generated'] };
    const st = T[d.verdict] || T.uncertain;
    let h = '<div class="img-src">🤖 ' + esc(d.model || 'AI-image detector') + '</div>';
    h += '<div class="cr-status ' + st[0] + '">' + st[1] + (d.confidence ? ' · ' + d.confidence + '% confident' : '') + '</div>';
    const chips = [];
    if (d.score != null) chips.push('AI likelihood: ' + d.score + '%');
    if (d.real_score != null) chips.push('Real likelihood: ' + d.real_score + '%');
    if (chips.length) h += '<div class="ppl-chips">' + chips.map(c => '<span class="ppl-chip">' + esc(c) + '</span>').join('') + '</div>';
    if (d.summary) h += '<p>' + esc(d.summary) + '</p>';
    if (d.reasons && d.reasons.length) h += title('🔍 Why I think this') + list(d.reasons);
    if (d.notes && d.notes.length) h += title('📝 Notes') + list(d.notes);
    return h;
  }
  const prevObj = window.__renderObjects;
  window.__renderObjects = function (d) {
    if (d.kind === 'medical') return renderMedical(d);
    if (d.kind === 'ocr') return renderOcr(d);
    if (d.kind === 'aidetect') return renderAiDetect(d);
    return prevObj ? prevObj(d) : '';
  };

  // ---------- chat + search panel ----------
  const panel = $('img-ask'), abox = $('ia-box'), inp = $('ia-input'), send = $('ia-send'), fileIn = $('img-file');
  if (!panel || !abox || !inp || !send || !fileIn) return;
  let imgId = null, imgFile = null, busy = false;
  const scroll = () => { abox.scrollTop = abox.scrollHeight; };
  const TYPING = '<span class="typing"><i></i><i></i><i></i></span>';

  function row(role) {
    const r = document.createElement('div'); r.className = 'msg-row ' + role;
    const a = document.createElement('div'); a.className = 'avatar'; a.textContent = role === 'user' ? 'U' : 'AI';
    const b = document.createElement('div'); b.className = 'bubble';
    r.appendChild(a); r.appendChild(b); abox.appendChild(r); scroll();
    return b;
  }
  async function pollJob(url, init) {
    const r = await fetch(url, init);
    const s = JSON.parse(await r.text());
    if (!s.job) throw new Error(s.error || 'Could not start.');
    let fails = 0;
    while (true) {
      await sleep(800);
      try {
        const p = JSON.parse(await (await fetch('/api/job/' + s.job)).text());
        fails = 0;
        if (p.done) return p.result;
      } catch (e) { if (++fails > 80) throw new Error('Lost connection to the server.'); }
    }
  }
  async function ensureId() {
    const f = fileIn.files[0];
    if (!f) throw new Error('Choose an image first.');
    if (imgId && imgFile === f) return imgId;
    const fd = new FormData(); fd.append('image', f);
    const r = await fetch('/api/image_upload', { method: 'POST', body: fd });
    const d = JSON.parse(await r.text());
    if (!d.success) throw new Error(d.error || 'Upload failed.');
    imgId = d.id; imgFile = f;
    return imgId;
  }
  function show(b, d) {
    let h = renderMarkdown(d.reply || '');
    if (d.image) h += '<img class="result" src="' + d.image + '">';
    if (d.text) h += box(d.text, 'Text from image');
    if (d.medical) h += renderMedical(d.medical);
    if (d.aidetect) h += renderAiDetect(d.aidetect);
    b.dataset.replyText = d.reply || '';
    b.dataset.replyLang = d.language || '';
    h += '<div class="oc-row" style="margin-top:4px"><button type="button" class="oc-btn ia-speak">🔊 Read aloud</button></div>';
    b.innerHTML = h;
  }
  abox.addEventListener('click', e => {
    const s = e.target.closest('.ia-speak'); if (!s) return;
    const bb = s.closest('.bubble');
    const txt = (bb && bb.dataset.replyText) || '';
    if (!txt || !window.speechSynthesis) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(txt.slice(0, 1500));
    const lg = bb.dataset.replyLang || '';
    u.lang = lg || navigator.language || 'en-US';
    speechSynthesis.speak(u);
  });
  async function ask(text) {
    const msg = (text || inp.value).trim();
    if (!msg || busy) return;
    busy = true; send.disabled = true;
    row('user').textContent = msg;
    inp.value = ''; inp.style.height = 'auto';
    const b = row('assistant'); b.innerHTML = TYPING;
    try {
      let res = null;
      for (let attempt = 0; attempt < 2 && !res; attempt++) {
        try {
          let id = '';
          if (fileIn.files[0]) id = await ensureId();
          res = await pollJob('/api/image_ask_start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                         body: JSON.stringify({ id: id, message: msg }) });
          if (!res.success) throw new Error(res.error || 'Something went wrong.');
        } catch (e) {
          res = null;
          if (attempt === 0 && /expired/i.test(e.message)) { imgId = null; continue; }
          throw e;
        }
      }
      show(b, res);
    } catch (e) { b.textContent = 'Error: ' + e.message; }
    scroll(); busy = false; send.disabled = false;
  }
  send.addEventListener('click', () => ask());
  inp.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });
  inp.addEventListener('input', () => { inp.style.height = 'auto'; inp.style.height = Math.min(inp.scrollHeight, 140) + 'px'; });
  panel.querySelectorAll('.ia-chip').forEach(c => c.addEventListener('click', () => ask(c.dataset.q)));
  fileIn.addEventListener('change', () => {
    imgId = null; abox.innerHTML = '';
    panel.style.display = 'block';
  });
})();
"""

_more_prev_get_html = WebUI.get_html


def _more_get_html(self) -> str:
    html = _more_prev_get_html(self)
    marker = '<div class="output" id="img-output"></div>'
    if marker in html:
        html = html.replace(marker, marker + MORE_PANEL, 1)
    else:
        print("⚠️ image tools patch: could not find the image output box, chat panel skipped")
    html = html.replace('</head>', '<style>' + MORE_CSS + '</style></head>', 1)
    return html.replace('</body>', '<script>' + MORE_JS + '</script></body>', 1)


WebUI.get_html = _more_get_html

# ============================================================================
# MATH STUDIO - ENGINE (typed / LaTeX / handwritten text -> exact answer)
# ============================================================================
import re
import json as _json
import threading
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr as _sp_parse, standard_transformations,
                                        implicit_multiplication_application, convert_xor)

_MX_T = standard_transformations + (implicit_multiplication_application, convert_xor)


class _MxDry(Exception):
    """Raised after parsing when only a syntax check was requested."""


# ---------------------------------------------------------------- namespace
def _mx_namespace():
    ns = {c: sp.Symbol(c) for c in "abcdfghijklmnopqrstuvwxyzABCDFGHJKLMNOPQRSTUVWXYZ"}
    for g in ("alpha", "beta", "theta", "lam", "mu", "sigma", "phi", "omega", "Delta", "delta", "epsilon"):
        ns[g] = sp.Symbol(g)
    ns.update({
        "e": sp.E, "E": sp.E, "I": sp.I, "pi": sp.pi, "oo": sp.oo, "inf": sp.oo,
        "sqrt": sp.sqrt, "cbrt": sp.cbrt, "root": sp.root, "exp": sp.exp,
        "ln": sp.log, "log": sp.log, "log10": lambda v: sp.log(v, 10), "log2": lambda v: sp.log(v, 2),
        "abs": sp.Abs, "Abs": sp.Abs, "sign": sp.sign, "floor": sp.floor, "ceiling": sp.ceiling,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "arcsin": sp.asin, "arccos": sp.acos,
        "arctan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "factorial": sp.factorial, "binomial": sp.binomial, "gamma": sp.gamma,
        "Min": sp.Min, "Max": sp.Max, "Matrix": sp.Matrix,
    })
    return ns


# ---------------------------------------------------------------- text cleaning
_SUP = {"²": "**2", "³": "**3", "⁴": "**4", "⁵": "**5", "⁶": "**6", "⁷": "**7", "⁸": "**8", "⁹": "**9", "¹": "**1", "⁰": "**0"}
_UNI = {"×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "π": "pi", "∞": "oo",
        "≤": "<=", "≥": ">=", "≠": "!=", "θ": "theta", "α": "alpha", "β": "beta", "λ": "lam", "μ": "mu",
        "σ": "sigma", "φ": "phi", "ω": "omega", "Δ": "Delta", "δ": "delta", "ε": "epsilon", "°": "*pi/180",
        "→": "->", "⟶": "->", "′": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " ", "∂": "d", "%": "/100"}


def _mx_unicode(s):
    for k, v in _SUP.items():
        s = s.replace(k, v)
    for k, v in _UNI.items():
        s = s.replace(k, v)
    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*([0-9A-Za-z.]+)", r"sqrt(\1)", s)
    s = re.sub(r"∛\s*\(", "cbrt(", s)
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    return s


def _mx_group(s, i):
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def _mx_arg(s, i):
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _mx_group(s, i)
    if s[i] == "\\":
        m = re.match(r"\\[a-zA-Z]+", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


_GREEK = {"alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "theta": "theta",
          "lambda": "lam", "mu": "mu", "sigma": "sigma", "phi": "phi", "omega": "omega", "epsilon": "epsilon",
          "Delta": "Delta"}


def _mx_tex2plain(s):
    s = s.strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    for a in ("\\(", "\\)", "\\[", "\\]"):
        s = s.replace(a, " ")
    s = re.sub(r"\\(left|right)(?![a-zA-Z])\.?", " ", s)
    s = re.sub(r"\\(displaystyle|quad|qquad|limits|nolimits)(?![a-zA-Z])", " ", s)
    # matrices and equation blocks
    s = re.sub(r"\\begin\{(?:p|b|v|B|V)?matrix\}(.*?)\\end\{(?:p|b|v|B|V)?matrix\}",
               lambda m: "Matrix([" + ",".join(
                   "[" + ",".join("(" + c.strip() + ")" for c in r.split("&")) + "]"
                   for r in re.split(r"\\\\", m.group(1)) if r.strip()) + "])", s, flags=re.S)
    s = re.sub(r"\\begin\{(cases|aligned|align\*?|array|gathered|split)\}(?:\{[^}]*\})?(.*?)\\end\{\1\}",
               lambda m: re.sub(r"\\\\", ";", m.group(2)).replace("&", ""), s, flags=re.S)
    s = s.replace("\\\\", ";")
    for a in ("\\,", "\\;", "\\!", "\\:"):
        s = s.replace(a, " ")
    s = re.sub(r"\\ ", " ", s)
    # derivative operators
    s = re.sub(r"\\[dt]?frac\s*\{\s*(?:d|\\partial)\s*(?:\^\{?(\d)\}?)?\s*\}\s*\{\s*(?:d|\\partial)\s*([a-zA-Z])\s*(?:\^\{?\d\}?)?\s*\}",
               lambda m: f"§DDX¦{m.group(2)}¦{m.group(1) or 1}§", s)
    # fractions
    for _ in range(60):
        m = re.search(r"\\[dt]?frac(?![a-zA-Z])", s)
        if not m:
            break
        a, i = _mx_arg(s, m.end())
        b, j = _mx_arg(s, i)
        s = s[:m.start()] + "((" + a + ")/(" + b + "))" + s[j:]
    # roots
    for _ in range(60):
        m = re.search(r"\\sqrt(?![a-zA-Z])", s)
        if not m:
            break
        i, n = m.end(), None
        if i < len(s) and s[i] == "[":
            j = s.find("]", i)
            if j > 0:
                n, i = s[i + 1:j], j + 1
        a, j = _mx_arg(s, i)
        s = s[:m.start()] + (f"root(({a}),({n}))" if n else f"sqrt({a})") + s[j:]
    # calculus markers
    bound = r"(\{[^{}]*\}|[^\s^{}\\]+|\\[a-zA-Z]+)"
    s = re.sub(r"\\int(?![a-zA-Z])(?:\s*_\s*" + bound + r")?(?:\s*\^\s*" + bound + r")?",
               lambda m: "§INT¦" + (m.group(1) or "").strip("{}") + "¦" + (m.group(2) or "").strip("{}") + "§", s)

    def _lim(m):
        arg = m.group(1)
        mm = re.match(r"\s*([a-zA-Z])\s*\\to\s*(.+?)\s*$", arg)
        if not mm:
            return " "
        var, pt, d = mm.group(1), mm.group(2), "+-"
        dm = re.search(r"\^\s*\{?\s*([+-])\s*\}?\s*$", pt)
        if dm:
            d, pt = dm.group(1), pt[:dm.start()]
        return f"§LIM¦{var}¦{pt.strip()}¦{d}§"
    s = re.sub(r"\\lim\s*_\s*\{([^{}]*)\}", _lim, s)
    s = re.sub(r"\\sum\s*_\s*\{\s*([a-zA-Z])\s*=\s*([^{}]*)\}\s*\^\s*(\{[^{}]*\}|[^\s{}]+)",
               lambda m: f"§SUM¦{m.group(1)}¦{m.group(2).strip()}¦{m.group(3).strip('{}')}§", s)
    s = re.sub(r"\\log\s*_\s*\{?(\w+)\}?\s*\(([^()]*)\)", r"log(\2,\1)", s)
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp|min|max)(?![a-zA-Z])", r"\1", s)
    for k, v in _GREEK.items():
        s = re.sub(r"\\" + k + r"(?![a-zA-Z])", v, s)
    for a, b in ((r"\\cdot", "*"), (r"\\times", "*"), (r"\\div", "/"), (r"\\pi", "pi"), (r"\\infty", "oo"),
                 (r"\\leq?", "<="), (r"\\geq?", ">="), (r"\\ne(?:q)?", "!="), (r"\\to", "->"),
                 (r"\\pm", "+"), (r"\\%", "/100"), (r"\\prime", "'")):
        s = re.sub(a + r"(?![a-zA-Z])", b, s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"_\s*\{([^{}]*)\}", lambda m: "_" + re.sub(r"\W", "", m.group(1)), s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    return s


def _mx_plain(s):
    """Any input (LaTeX / words / unicode) -> plain text with §markers§ for calculus."""
    if "\\" in s or re.search(r"\^\{|_\{", s):
        s = _mx_tex2plain(s)
    s = _mx_unicode(s)
    s = s.replace("\r", "\n")
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|ln)\s*(?:\^|\*\*)\s*\(?(\d+)\)?\s*(\([^()]*\)|[a-z0-9.]+)",
               r"(\1(\3))**\2", s)
    s = re.sub(r"∫\s*(?:_\s*\(?([^\s^()]+)\)?)?\s*(?:\^\s*\(?([^\s()]+)\)?)?",
               lambda m: "§INT¦" + (m.group(1) or "") + "¦" + (m.group(2) or "") + "§", s)
    s = re.sub(r"Σ\s*_?\(?\s*([a-z])\s*=\s*([^\s^()]+)\)?\s*\^\(?([^\s()]+)\)?",
               lambda m: f"§SUM¦{m.group(1)}¦{m.group(2)}¦{m.group(3)}§", s)
    m = re.match(r"^\s*lim(?:it)?\s*(?:as\s*)?_?\(?\s*([a-z])\s*(?:->|to)\s*([^\s)]+)\s*\)?\s*(?:of\s+)?(.+)$", s, re.I | re.S)
    if m:
        pt, d = m.group(2), "+-"
        dm = re.match(r"^(.+?)\^?([+-])$", pt)
        if dm and not re.fullmatch(r"[+-]", dm.group(1)):
            pt, d = dm.group(1), dm.group(2)
        s = f"§LIM¦{m.group(1)}¦{pt}¦{d}§ {m.group(3)}"
    m = re.match(r"^\s*d\^?(\d)?\s*/\s*d([a-z])\^?\d?\s*(.*)$", s, re.S)
    if m:
        s = f"§DDX¦{m.group(2)}¦{m.group(1) or 1}§ {m.group(3)}"
    return s


_WORDS = [
    (r"^(?:please\s+)?(?:solve|find\s+x|find\s+the\s+roots?\s+of|roots?\s+of)\b[:\s]*", "solve"),
    (r"^(?:simplify|reduce)\b[:\s]*", "simplify"),
    (r"^expand\b[:\s]*", "expand"),
    (r"^factori[sz]e\b[:\s]*|^factor\b[:\s]*", "factor"),
    (r"^(?:differentiate|derivative\s+of|diff)\b[:\s]*", "derivative"),
    (r"^(?:integrate|integral\s+of|antiderivative\s+of)\b[:\s]*", "integral"),
    (r"^(?:evaluate|calculate|compute|what\s+is)\b[:\s]*", "evaluate"),
    (r"^(?:plot|graph)\b[:\s]*", "plot"),
]


def _mx_words(raw):
    s, hint = raw.strip(), ""
    if not s.startswith("\\"):
        for rx, op in _WORDS:
            m = re.match(rx, s, re.I)
            if m:
                s, hint = s[m.end():], op
                break
    s = re.sub(r"\s*=\s*\?*\s*$", "", s)
    return s.strip(), hint


# ---------------------------------------------------------------- parsing
def _mx_fixup(s):
    s = s.strip().rstrip(".,")
    if re.match(r"^\[\s*\[", s):
        s = "Matrix(" + s + ")"
    if "Matrix" not in s:
        s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"(?<![A-Za-z0-9_])([A-Za-z])\s*\(", r"\1*(", s)       # x(x+1) means x*(x+1)
    return s


def _mx_parse(s, ns=None):
    s = _mx_fixup(s)
    if not s:
        raise ValueError("One side of the equation is empty.")
    try:
        return _sp_parse(s, local_dict=ns or _mx_namespace(), transformations=_MX_T, evaluate=True)
    except Exception:
        raise ValueError(f"I could not understand \u201c{s}\u201d. Check the brackets and operators, or edit the text.")


def _mx_split_top(s, seps=";,\n"):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch in seps and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [p.strip() for p in out if p.strip()]


def _mx_rel(s):
    parts = re.split(r"(<=|>=|!=|==|=|<|>)", s)
    if len(parts) == 3:
        return parts[0].strip(), parts[1], parts[2].strip()
    return None


def _mx_pick(syms, var=""):
    syms = list(syms)
    if var:
        v = sp.Symbol(var.strip())
        return v
    for pref in "xtyzuvnrs":
        for s_ in syms:
            if str(s_) == pref:
                return s_
    return syms[0] if syms else sp.Symbol("x")


def _mx_txt(o):
    t = str(o).replace("**", "^").replace("sqrt", "√")
    return re.sub(r"(?<![A-Za-z_])oo(?![A-Za-z_])", "∞", t)


def _mx_set_txt(S):
    if isinstance(S, sp.Union):
        return " ∪ ".join(_mx_set_txt(a) for a in S.args)
    if isinstance(S, sp.Interval):
        l_, r_ = ("(" if S.left_open else "["), (")" if S.right_open else "]")
        return f"{l_}{_mx_txt(S.start)}, {_mx_txt(S.end)}{r_}"
    if isinstance(S, sp.FiniteSet):
        return "{" + ", ".join(_mx_txt(a) for a in S.args) + "}"
    if isinstance(S, sp.ImageSet):
        lam = S.lamda
        return "{" + _mx_txt(lam.expr).replace("_", "") + "  for every whole number " + str(lam.variables[0]).replace("_", "") + "}"
    return _mx_txt(S)


def _mx_num(o, d=8):
    try:
        v = sp.N(o, d)
        return str(v)
    except Exception:
        return ""


def _mx_is_exact_simple(o):
    return o.is_Integer or o.is_Rational and False


def _mx_with_approx(o):
    t = _mx_txt(o)
    try:
        if o.is_number and o.is_real and not o.is_Integer and not o.is_Float:
            a = sp.N(o, 6)
            if str(a) != t:
                t += f"  ≈ {a}"
    except Exception:
        pass
    return t


def _mx_timeout(fn, sec=25):
    box = {}

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:
            box["e"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(sec)
    if t.is_alive():
        raise TimeoutError(f"This took longer than {sec} seconds to compute. Try a simpler form, or give bounds for an integral.")
    if "e" in box:
        raise box["e"]
    return box["v"]


# ---------------------------------------------------------------- plotting data
def _mx_plot(series, x, marks=None, xr=None):
    try:
        import numpy as np
    except Exception:
        return None
    lo, hi = xr or (-10.0, 10.0)
    xs = np.linspace(lo, hi, 360)
    out = []
    for label, expr in series:
        try:
            f = sp.lambdify(x, expr, "numpy")
            ys = np.asarray(f(xs))
            if ys.shape == ():
                ys = np.full_like(xs, float(ys))
            if np.iscomplexobj(ys):
                ys = np.where(np.abs(ys.imag) < 1e-9, ys.real, np.nan)
            ys = ys.astype(float)
            ys[~np.isfinite(ys)] = np.nan
            out.append({"label": label, "pts": [[float(a), (None if np.isnan(b) else float(b))] for a, b in zip(xs, ys)]})
        except Exception:
            continue
    if not out:
        return None
    return {"series": out, "marks": [float(m) for m in (marks or [])], "xmin": float(lo), "xmax": float(hi), "var": str(x)}


def _mx_range(roots):
    try:
        r = [float(v) for v in roots if v.is_real and abs(float(v)) < 1e6]
    except Exception:
        r = []
    if not r:
        return (-10.0, 10.0)
    a, b = min(r), max(r)
    pad = max(3.0, (b - a) * 0.5)
    return (a - pad, b + pad)


def _mx_numeric_roots(f, x, lo=-50, hi=50):
    import numpy as np
    g = sp.lambdify(x, f, "numpy")
    xs = np.linspace(lo, hi, 4001)
    with np.errstate(all="ignore"):
        ys = np.asarray(g(xs), dtype=float)
    roots = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b <= 0 and a != b:
            try:
                r = sp.nsolve(f, x, (xs[i], xs[i + 1]), solver="bisect")
            except Exception:
                r = sp.Float((xs[i] + xs[i + 1]) / 2)
            try:
                if abs(complex(f.subs(x, r))) < 1e-6 and all(abs(float(r) - float(q)) > 1e-6 for q in roots):
                    roots.append(r)
            except Exception:
                pass
    return roots


# ---------------------------------------------------------------- result helpers
def _st(steps, text, latex=None):
    steps.append({"text": text, "latex": latex})


def _res(op, inp, ans_tex, ans_txt, steps, **kw):
    r = {"success": True, "operation": op, "input_latex": inp, "answer_latex": ans_tex,
         "answer_text": ans_txt, "steps": steps, "extra": [], "plot": None, "notes": []}
    r.update(kw)
    return r


_mxL = sp.latex


# ---------------------------------------------------------------- the operations
def _mx_equation(lhs, rhs, var, steps, notes):
    f = sp.expand(lhs - rhs)
    syms = sorted((lhs - rhs).free_symbols, key=str)
    inp = _mxL(sp.Eq(lhs, rhs))
    if not syms:
        ok = bool(sp.simplify(lhs - rhs) == 0)
        _st(steps, "Both sides are numbers, so I just compare them.", f"{_mxL(lhs)} \\;?\\; {_mxL(rhs)}")
        return _res("Check equality", inp, r"\text{True}" if ok else r"\text{False}",
                    "True" if ok else "False", steps)
    x = _mx_pick(syms, var)
    if len(syms) > 1:
        notes.append(f"There is more than one letter, so I solved for {x}. Type the variable in the Variable box to solve for another one.")
    _st(steps, f"Move everything to one side and solve for {x}.", f"{_mxL(f)} = 0")
    sols, set_ans, complex_sols, poly = [], None, [], False
    try:
        P = sp.Poly(f, x)
        poly = P.degree() >= 1 and all(not c.has(x) for c in P.all_coeffs())
        if poly:
            fac = sp.factor(f)
            if fac != f:
                _st(steps, "Factor it.", f"{_mxL(fac)} = 0")
            allr = sp.solve(f, x)
            sols = [r for r in allr if r.is_real]
            complex_sols = [r for r in allr if r.is_real is False or (r.is_real is None and not r.is_number)]
    except Exception:
        poly = False
    if not poly:
        ss = sp.solveset(sp.Eq(lhs, rhs), x, domain=sp.S.Reals)
        if isinstance(ss, sp.FiniteSet):
            sols = list(ss)
        elif ss == sp.S.EmptySet:
            sols = []
        elif isinstance(ss, sp.ConditionSet) or ss.has(sp.ConditionSet):
            nr = []
            try:
                nr = _mx_numeric_roots(f, x)
            except Exception:
                pass
            if nr:
                sols = nr
                notes.append("No exact formula exists, so these are numerical solutions between -50 and 50.")
            else:
                set_ans = ss
        else:
            set_ans = ss
    try:
        sols = sorted(sols, key=lambda r: float(sp.re(r)) if r.is_number else 0)
    except Exception:
        pass
    if set_ans is not None:
        _st(steps, "The solutions repeat, so this is a general family of solutions.", f"{x} \\in {_mxL(set_ans)}")
        rng = (-10.0, 10.0)
        plot = _mx_plot([(f"{_mx_txt(f)}", f)], x, [], rng) if f.free_symbols == {x} else None
        r = _res("Solve equation", inp, f"{x} \\in {_mxL(set_ans)}", f"{x} ∈ {_mx_set_txt(set_ans)}", steps, plot=plot)
        return r
    if not sols and not complex_sols:
        _st(steps, "No value makes the equation true.")
        return _res("Solve equation", inp, r"\text{No real solution}", "No real solution", steps,
                    plot=_mx_plot([(_mx_txt(f), f)], x) if f.free_symbols == {x} else None)
    if not sols and complex_sols:
        notes.append("There are no real solutions. The solutions below are complex numbers.")
        sols, complex_sols = complex_sols, []
    for r_ in sols[:8]:
        try:
            lv, rv = sp.N(lhs.subs(x, r_), 8), sp.N(rhs.subs(x, r_), 8)
            if r_.free_symbols:
                continue
            ok = abs(complex(lv - rv)) < 1e-6 * (1 + abs(complex(rv)))
            _st(steps, f"Check {x} = {_mx_txt(r_)}:  left side = {lv}, right side = {rv}  {'✓' if ok else '✗'}")
        except Exception:
            pass
    tex = ",\\; ".join(f"{x} = {_mxL(r_)}" for r_ in sols)
    txt = ",   ".join(f"{x} = {_mx_with_approx(r_)}" for r_ in sols)
    extra = []
    if complex_sols:
        extra.append({"label": "Complex solutions", "latex": ",\\; ".join(f"{x} = {_mxL(r_)}" for r_ in complex_sols)})
    plot = None
    if f.free_symbols == {x}:
        plot = _mx_plot([(_mx_txt(f), f)], x, [r_ for r_ in sols if r_.is_real], _mx_range(sols))
    return _res("Solve equation", inp, tex, txt, steps, extra=extra, plot=plot)


def _mx_system(eqs, var, steps, notes):
    vars_ = sorted(set().union(*[(a - b).free_symbols for a, b in eqs]), key=str)
    if not vars_:
        raise ValueError("There are no unknown letters in these equations.")
    inp = r"\begin{cases}" + r"\\".join(_mxL(sp.Eq(a, b)) for a, b in eqs) + r"\end{cases}"
    _st(steps, f"Solve all {len(eqs)} equations together for {', '.join(map(str, vars_))}.")
    sol = sp.solve([sp.Eq(a, b) for a, b in eqs], vars_, dict=True)
    if not sol:
        _st(steps, "The equations contradict each other.")
        return _res("Solve system", inp, r"\text{No solution}", "No solution", steps)
    tex, txt = [], []
    for d in sol:
        tex.append(",\\; ".join(f"{k} = {_mxL(v)}" for k, v in d.items()))
        txt.append(",  ".join(f"{k} = {_mx_with_approx(v)}" for k, v in d.items()))
        try:
            allok = all(sp.simplify((a - b).subs(d)) == 0 for a, b in eqs)
            _st(steps, "Check: put the solution back into every equation  " + ("✓" if allok else "✗"))
        except Exception:
            pass
    free = set().union(*[v.free_symbols for d in sol for v in d.values()])
    if free:
        notes.append("There are infinitely many solutions: " + ", ".join(map(str, free)) + " can be any number.")
    return _res("Solve system", inp, " \\quad\\text{or}\\quad ".join(tex), "   or   ".join(txt), steps)


def _mx_inequality(lhs, op, rhs, var, steps, notes):
    rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op](lhs, rhs)
    syms = sorted(rel.free_symbols, key=str)
    if not syms:
        raise ValueError("There is no unknown letter in this inequality.")
    x = _mx_pick(syms, var)
    _st(steps, f"Find all values of {x} that make this true.", _mxL(rel))
    try:
        sol = sp.solve_univariate_inequality(rel, x, relational=False)
    except Exception:
        sol = sp.solveset(rel, x, domain=sp.S.Reals)
    f = lhs - rhs
    try:
        crit = sp.solveset(sp.Eq(f, 0), x, domain=sp.S.Reals)
        if isinstance(crit, sp.FiniteSet) and crit:
            _st(steps, "The sign can only change at these points:", ", ".join(f"{x}={_mxL(c)}" for c in crit))
    except Exception:
        crit = []
    plot = _mx_plot([(_mx_txt(f), f)], x, list(crit) if isinstance(crit, sp.FiniteSet) else [],
                    _mx_range(list(crit)) if isinstance(crit, sp.FiniteSet) else None) if f.free_symbols == {x} else None
    return _res("Solve inequality", _mxL(rel), f"{x} \\in {_mxL(sol)}", f"{x} ∈ {_mx_set_txt(sol)}", steps, plot=plot)


def _mx_ode(s, steps, notes):
    t = sp.Symbol("x")
    m = re.search(r"\b([a-zA-Z])('+)", s)
    fn = m.group(1) if m else ("y" if re.search(r"\bd\^?\d?y\s*/", s) else "y")
    s = re.sub(rf"\b{fn}\(x\)", fn, s)
    s = re.sub(rf"\bd\^?2{fn}\s*/\s*dx\^?2", "Dy2", s)
    s = re.sub(rf"\bd{fn}\s*/\s*dx", "Dy1", s)
    s = re.sub(rf"\b{fn}'''", "Dy3", s)
    s = re.sub(rf"\b{fn}''", "Dy2", s)
    s = re.sub(rf"\b{fn}'", "Dy1", s)
    yf = sp.Function(fn)
    ns = _mx_namespace()
    ns.update({"x": t, fn: yf(t), "Dy1": sp.Derivative(yf(t), t), "Dy2": sp.Derivative(yf(t), (t, 2)),
               "Dy3": sp.Derivative(yf(t), (t, 3))})
    pieces = _mx_split_top(s, ";\n")
    ics = {}
    eq = None
    for p in pieces:
        r = _mx_rel(p)
        if not r:
            continue
        lhs = _mx_parse(r[0], ns)
        rhs = _mx_parse(r[2], ns)
        if lhs.has(sp.Derivative) or rhs.has(sp.Derivative):
            eq = sp.Eq(lhs, rhs)
    if eq is None:
        raise ValueError("I could not find a differential equation (an equation with y' or dy/dx).")
    _st(steps, "Solve the differential equation for " + fn + "(x).", _mxL(eq))
    sol = sp.dsolve(eq, yf(t))
    if isinstance(sol, list):
        tex = ",\\quad ".join(_mxL(q) for q in sol)
        txt = ";  ".join(f"{_mx_txt(q.lhs)} = {_mx_txt(q.rhs)}" for q in sol)
    else:
        tex, txt = _mxL(sol), f"{_mx_txt(sol.lhs)} = {_mx_txt(sol.rhs)}"
    notes.append("C1, C2 are constants. They are fixed by starting values such as y(0) = 1.")
    return _res("Differential equation", _mxL(eq), tex, txt, steps)


def _mx_matrix(M, op, steps, notes):
    if not isinstance(M, sp.MatrixBase):
        raise ValueError("That is not a matrix.")
    rows = []
    items = [("Matrix", M), ("Transpose", M.T)]
    if M.is_square:
        items += [("Determinant", M.det()), ("Rank", M.rank())]
        if M.det() != 0:
            items.append(("Inverse", M.inv()))
        else:
            _st(steps, "The determinant is 0, so there is no inverse.")
        try:
            items.append(("Eigenvalues", M.eigenvals()))
        except Exception:
            pass
    else:
        items.append(("Rank", M.rank()))
    items.append(("Row-reduced form", M.rref()[0]))
    for name, v in items:
        if isinstance(v, dict):
            tex = ",\\; ".join(f"{_mxL(k)}\\ (\\times{m_})" for k, m_ in v.items())
        else:
            tex = _mxL(v)
        rows.append({"label": name, "latex": tex})
    main = next((r for r in rows if r["label"] == "Determinant"), rows[0])
    return _res("Matrix", _mxL(M), main["latex"], ("determinant = " + _mx_txt(M.det())) if M.is_square else ("rank = " + str(M.rank())), steps, extra=rows[1:])


def mx_solve(text, op="auto", var="", lo="", hi="", point="", order="", dry=False):
    try:
        return _mx_solve(text, op, var, lo, hi, point, order, dry)
    except _MxDry:
        return {"success": True, "dry": True}


def _mx_solve(text, op, var, lo, hi, point, order, dry):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Write or type an equation first.")
    steps, notes = [], []
    s, hint = _mx_words(raw)
    if op in ("", None, "auto"):
        op = hint or "auto"
    s = _mx_plain(s)
    if re.search(r"\blog\(([^,()]*)\)", s):
        notes.append("log is read as the natural logarithm (same as ln). Write log10(x) for base 10.")
    ns = _mx_namespace()

    def P(t):
        return _mx_parse(t, ns)

    def done():
        if dry:
            raise _MxDry()

    # ---- markers (integral, limit, sum, derivative) ----
    m_int = re.search(r"§INT¦(.*?)¦(.*?)§", s)
    m_lim = re.search(r"§LIM¦(.*?)¦(.*?)¦(.*?)§", s)
    m_sum = re.search(r"§SUM¦(.*?)¦(.*?)¦(.*?)§", s)
    m_ddx = re.search(r"§DDX¦(.*?)¦(.*?)§", s)
    if m_int and op in ("auto", "integral"):
        op, lo, hi = "integral", lo or m_int.group(1), hi or m_int.group(2)
        s = s[m_int.end():]
    elif m_lim and op in ("auto", "limit"):
        op, var, point = "limit", var or m_lim.group(1), point or m_lim.group(2)
        order = order or ("" if m_lim.group(3) == "+-" else m_lim.group(3))
        s = s[m_lim.end():]
    elif m_sum and op in ("auto", "sum"):
        op, var, lo, hi = "sum", var or m_sum.group(1), lo or m_sum.group(2), hi or m_sum.group(3)
        s = s[m_sum.end():]
    elif m_ddx and op in ("auto", "derivative"):
        op, var, order = "derivative", var or m_ddx.group(1), order or m_ddx.group(2)
        s = s[m_ddx.end():]
    s = s.strip()

    # ---- auto-detect the kind of problem ----
    if op == "auto":
        if "Matrix" in s or re.match(r"^\s*\[\s*\[", s):
            op = "matrix"
        elif re.search(r"[a-zA-Z]'", s) or re.search(r"\bd\^?\d?[a-z]\s*/\s*d[a-z]", s) and "=" in s:
            op = "ode"
        else:
            pieces = _mx_split_top(s)
            rels = [_mx_rel(p) for p in pieces]
            if len(pieces) > 1 and all(r and r[1] == "=" for r in rels):
                op = "system"
            elif len(pieces) == 1 and rels[0]:
                op = "inequality" if rels[0][1] in ("<", "<=", ">", ">=") else ("solve" if rels[0][1] in ("=", "==") else "inequality_ne")
            else:
                op = "expr"

    if op == "ode":
        done()
        return _mx_timeout(lambda: _mx_ode(s, steps, notes))
    if op == "matrix":
        M = P(s)
        done()
        return _mx_timeout(lambda: _mx_matrix(M, op, steps, notes))

    if op == "system":
        eqs = []
        for p in _mx_split_top(s):
            r = _mx_rel(p)
            if not r:
                raise ValueError(f"“{p}” has no = sign. Separate equations with commas.")
            eqs.append((P(r[0]), P(r[2])))
        done()
        res = _mx_timeout(lambda: _mx_system(eqs, var, steps, notes))
        res["notes"] = notes
        return res

    if op in ("solve", "inequality", "inequality_ne"):
        r = _mx_rel(s)
        if not r:
            # "solve x^2-4" means "= 0"
            r = (s, "=", "0")
        lhs, rel, rhs = P(r[0]), r[1], P(r[2])
        done()
        if rel in ("=", "=="):
            res = _mx_timeout(lambda: _mx_equation(lhs, rhs, var, steps, notes))
        elif rel == "!=":
            raise ValueError("≠ is not supported for solving yet.")
        else:
            res = _mx_timeout(lambda: _mx_inequality(lhs, rel, rhs, var, steps, notes))
        res["notes"] = notes + res.get("notes", [])
        return res

    # ---- single expression operations ----
    if op == "integral" and not (m_int):
        mf = re.search(r"\bfrom\s+(\S+)\s+to\s+(\S+)\s*$", s, re.I)
        if mf:
            lo, hi, s = lo or mf.group(1), hi or mf.group(2), s[:mf.start()]
    if op == "integral":
        md = re.match(r"^(.*?)\s*(?<![A-Za-z])d\s*([a-z])\s*$", s.strip(), re.S)
        if md:
            s, var = md.group(1), var or md.group(2)
    if op == "plot" and "=" in s:
        a, b = s.split("=", 1)
        s = f"({a})-({b})"
    expr = P(s)
    free = sorted(expr.free_symbols, key=str) if hasattr(expr, "free_symbols") else []
    x = _mx_pick(free, var) if (free or var) else sp.Symbol("x")
    inp = _mxL(expr)
    done()

    def work():
        nonlocal op
        if op == "derivative":
            n = int(order or 1)
            r = sp.diff(expr, x, n)
            rs = sp.simplify(r)
            tex_op = f"\\frac{{d{'^%d' % n if n > 1 else ''}}}{{d{x}{'^%d' % n if n > 1 else ''}}}"
            _st(steps, f"Differentiate with respect to {x}" + (f", {n} times." if n > 1 else "."), f"{tex_op}\\left({inp}\\right)")
            if rs != r:
                _st(steps, "Raw result, then simplified.", f"{_mxL(r)} \\;\\to\\; {_mxL(rs)}")
            ans = [("Derivative", rs)]
            extra = []
            if point:
                pv = P(point)
                v = sp.simplify(rs.subs(x, pv))
                extra.append({"label": f"Value at {x} = {point}", "latex": _mxL(v)})
            return _res("Derivative", inp, _mxL(rs), _mx_txt(rs), steps, extra=extra,
                        plot=_mx_plot([(_mx_txt(expr), expr), (_mx_txt(rs), rs)], x) if free and len(free) == 1 else None)
        if op == "integral":
            if lo != "" and hi != "":
                a, b = P(str(lo)), P(str(hi))
                _st(steps, f"Definite integral of the function from {x} = {lo} to {x} = {hi}.",
                    f"\\int_{{{_mxL(a)}}}^{{{_mxL(b)}}} {inp}\\,d{x}")
                r = sp.integrate(expr, (x, a, b))
                if r.has(sp.Integral):
                    r = sp.Integral(expr, (x, a, b)).evalf(15)
                    notes.append("There is no closed form, so this is a numerical value.")
                r = sp.simplify(r)
                _st(steps, "Antiderivative, then subtract the values at the bounds.")
                return _res("Definite integral", f"\\int_{{{_mxL(a)}}}^{{{_mxL(b)}}} {inp}\\,d{x}", _mxL(r), _mx_with_approx(r), steps,
                            plot=_mx_plot([(_mx_txt(expr), expr)], x, [], (min(float(a), float(b)) - 2, max(float(a), float(b)) + 2)) if (a.is_real and b.is_real and len(free) == 1) else None)
            r = sp.integrate(expr, x)
            if r.has(sp.Integral):
                notes.append("This integral has no formula made of standard functions.")
            _st(steps, f"Find a function whose derivative is the given one (integrate with respect to {x}).", f"\\int {inp}\\,d{x}")
            try:
                chk = sp.simplify(sp.diff(r, x) - expr) == 0
                _st(steps, "Check: differentiate the result and get the original back  " + ("✓" if chk else "✗"))
            except Exception:
                pass
            return _res("Indefinite integral", f"\\int {inp}\\,d{x}", _mxL(r) + " + C", _mx_txt(r) + " + C", steps)
        if op == "limit":
            pt = P(str(point or "0"))
            d = order if order in ("+", "-") else "+-"
            r = sp.limit(expr, x, pt, d)
            _st(steps, f"Find what the expression approaches as {x} → {point or 0}" + (f" from the {'right' if d == '+' else 'left'}." if d != "+-" else "."))
            return _res("Limit", f"\\lim_{{{x}\\to {_mxL(pt)}{'^' + d if d != '+-' else ''}}} {inp}", _mxL(r), _mx_with_approx(r), steps)
        if op == "sum":
            a, b = P(str(lo or 1)), P(str(hi or "oo"))
            r = sp.summation(expr, (x, a, b))
            _st(steps, f"Add the terms for {x} from {_mxL(a)} to {_mxL(b)}.")
            return _res("Sum", f"\\sum_{{{x}={_mxL(a)}}}^{{{_mxL(b)}}} {inp}", _mxL(sp.simplify(r)), _mx_with_approx(sp.simplify(r)), steps)
        if op == "series":
            n = int(order or 6)
            at = P(str(point or 0))
            r = sp.series(expr, x, at, n).removeO()
            _st(steps, f"Taylor series around {x} = {_mxL(at)} up to order {n - 1}.")
            return _res("Series", inp, _mxL(r) + " + \\cdots", _mx_txt(r) + " + …", steps)
        if op == "plot":
            return _res("Graph", inp, inp, _mx_txt(expr), steps, plot=_mx_plot([(_mx_txt(expr), expr)], x))
        # simplify / expand / factor / evaluate / auto-expression
        if not free:
            exact = sp.simplify(expr)
            approx = sp.N(exact, 15)
            _st(steps, "Calculate.", f"{inp} = {_mxL(exact)}")
            extra = [] if exact.is_Integer else [{"label": "Decimal", "latex": _mxL(sp.N(exact, 12))}]
            return _res("Calculate", inp, _mxL(exact) + ("" if exact.is_Integer or str(approx) == str(exact) else f" \\approx {_mxL(sp.N(exact, 10))}"),
                        _mx_with_approx(exact), steps, extra=extra)
        fns = {"simplify": sp.simplify, "expand": sp.expand, "factor": sp.factor, "evaluate": sp.simplify,
               "together": sp.together, "apart": lambda e: sp.apart(e, x), "auto": sp.simplify, "expr": sp.simplify}
        main = fns.get(op, sp.simplify)(expr)
        extra = []
        if op in ("expr", "auto", "simplify"):
            for nm, fn_ in (("Expanded", sp.expand), ("Factored", sp.factor)):
                try:
                    v = fn_(expr)
                    if v != main and v != expr:
                        extra.append({"label": nm, "latex": _mxL(v)})
                except Exception:
                    pass
            try:
                ts = sp.trigsimp(expr)
                if ts != main and expr.has(sp.sin, sp.cos, sp.tan):
                    extra.append({"label": "Trig-simplified", "latex": _mxL(ts)})
            except Exception:
                pass
        _st(steps, {"expand": "Multiply everything out.", "factor": "Write it as a product.", "apart": "Split into partial fractions."}.get(op, "Simplify."),
            f"{inp} = {_mxL(main)}")
        pl = _mx_plot([(_mx_txt(expr), expr)], x) if len(free) == 1 else None
        return _res("Simplify" if op in ("expr", "auto", "simplify", "evaluate") else op.title(), inp, _mxL(main), _mx_txt(main), steps, extra=extra, plot=pl)

    res = _mx_timeout(work)
    res["notes"] = notes + res.get("notes", [])
    return res


# ---------------------------------------------------------------- reading JSON from the AI
def _mx_json(txt):
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        d = _json.loads(t, strict=False)
    except Exception:
        try:
            d = _json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", t), strict=False)
        except Exception:
            m = re.search(r"\{.*\}", t, re.S)
            if not m:
                raise
            d = _json.loads(m.group(0), strict=False)

    def fix(v):                      # \frac, \beta, \times, \right must not become control characters
        if isinstance(v, str):
            return v.replace("\x0c", "\\f").replace("\x08", "\\b").replace("\t", "\\t").replace("\r", "\\r")
        return v
    return {k: fix(v) for k, v in d.items()}


def mx_parses(text):
    try:
        mx_solve(text, dry=True)
        return True
    except Exception:
        return False


def mx_same(a, b):
    """True if two recognised strings mean the same maths."""
    try:
        na = re.sub(r"\s+", "", _mx_plain(_mx_words(a)[0]))
        nb = re.sub(r"\s+", "", _mx_plain(_mx_words(b)[0]))
        if na == nb:
            return True
        ra, rb = _mx_rel(na), _mx_rel(nb)
        ns = _mx_namespace()
        if ra and rb and ra[1] == rb[1]:
            d = _mx_parse(ra[0], ns) - _mx_parse(ra[2], ns) - (_mx_parse(rb[0], ns) - _mx_parse(rb[2], ns))
            return sp.simplify(d) == 0 or sp.simplify(_mx_parse(ra[0], ns) - _mx_parse(ra[2], ns) + _mx_parse(rb[0], ns) - _mx_parse(rb[2], ns)) == 0
        if not ra and not rb:
            return sp.simplify(_mx_parse(na, ns) - _mx_parse(nb, ns)) == 0
    except Exception:
        pass
    return False
# ===== END ENGINE =====

# ============================================================================
# MATH STUDIO - HANDWRITING / IMAGE READERS + ROUTES + PAGE
# ============================================================================
from PIL import Image as _MxImage, ImageOps as _MxOps
import io as _mx_io

MATH_OCR_PROMPT = r"""You are an expert at reading handwritten and printed mathematics from images (notebooks, whiteboards, textbooks, screenshots, a finger or mouse drawing).
Transcribe EXACTLY what is written. Do NOT solve it and do NOT correct it.
Return ONLY a JSON object:
{"latex": "the equation or expression in standard LaTeX",
 "kind": "equation | expression | system | calculus | matrix | word_problem | other",
 "problem_text": "only for a word problem: the full problem text in its own language, otherwise an empty string",
 "confidence": 0-100,
 "notes": "short note about anything unclear, or an empty string"}
Rules:
- Several equations that belong together: use \begin{cases} ... \\ ... \end{cases}.
- Integrals, limits, sums and derivatives must keep their bounds and the variable (\int_0^1 x^2\,dx, \lim_{x\to 0}, \frac{d}{dx}).
- Use context to tell similar shapes apart: x and \times, 1 and l, 0 and O, 2 and z, 5 and S, + and t, - and an equals sign.
- Keep what is written even if it looks wrong. If a character is unreadable write ? there.
- If the image has no mathematics, return an empty latex and say so in notes."""

MATH_EXPLAIN_PROMPT = """You are a patient maths teacher. A computer algebra system has ALREADY solved the problem, so the final answer below is correct. Explain how to get to it.
Problem: __PROBLEM__
Verified answer: __ANSWER__
Write a clear step-by-step solution a student can follow: name the method, show each step on its own line, and end by stating the answer. Write maths in LaTeX between single dollar signs, for example $x^2-4=0$. Keep it short (at most about 12 lines). Reply in the same language as the problem (English if unsure). Do not contradict the verified answer."""


def _mx_clean_tex(t):
    t = (t or "").strip()
    t = re.sub(r"^```(?:latex|tex)?\s*|\s*```$", "", t)
    t = re.sub(r"^\\\[|\\\]$|^\$\$|\$\$$|^\$|\$$", "", t).strip()
    t = re.sub(r"\\begin\{equation\*?\}|\\end\{equation\*?\}", "", t).strip()
    return t.rstrip(".").strip()


class MathReader:
    """Reads maths from an image. Gemini vision first, Hugging Face / GitHub models as backup or for comparison.
    Local models: Pix2Text (GitHub breezedeus/pix2text), GOT-OCR 2.0 (Hugging Face), pix2tex (GitHub lukas-blecher/LaTeX-OCR)."""

    def __init__(self, chatbot):
        self.bot = chatbot
        self._p2t = self._tex = self._got = None
        self._lock = threading.Lock()

    # ---------- local engines (loaded the first time they are needed) ----------
    def _gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _read_p2t(self, img):
        if self._p2t is None:
            from pix2text import Pix2Text
            try:
                self._p2t = Pix2Text.from_config(enable_formula=True, enable_table=False,
                                                 device="cuda" if self._gpu() else "cpu")
            except TypeError:
                self._p2t = Pix2Text.from_config()
        out = self._p2t.recognize_formula(img)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        if isinstance(out, dict):
            out = out.get("text", "")
        return str(out)

    def _read_got(self, img):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if self._got is None:
            name = os.getenv("MATH_GOT_MODEL", "stepfun-ai/GOT-OCR-2.0-hf")
            gpu = self._gpu()
            proc = AutoProcessor.from_pretrained(name)
            model = AutoModelForImageTextToText.from_pretrained(
                name, torch_dtype=torch.float16 if gpu else torch.float32, device_map="cuda" if gpu else None)
            model.eval()
            self._got = (proc, model)
        proc, model = self._got
        inputs = proc(img, return_tensors="pt", format=True).to(model.device, model.dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, tokenizer=proc.tokenizer,
                                 stop_strings="<|im_end|>", max_new_tokens=1024)
        return proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _read_pix2tex(self, img):
        if self._tex is None:
            from pix2tex.cli import LatexOCR
            self._tex = LatexOCR()
        return str(self._tex(img))

    # ---------- main ----------
    def read(self, img, mode="auto"):
        cands, errors, meta = [], [], {"kind": "", "problem_text": "", "notes": ""}

        def add(engine, latex, conf=None):
            latex = _mx_clean_tex(latex)
            if latex:
                cands.append({"engine": engine, "latex": latex, "confidence": conf, "parses": mx_parses(latex)})

        if mode != "local" and self.bot is not None:
            try:
                d = _mx_json(self.bot.vision_json(_to_jpeg(img, 95), MATH_OCR_PROMPT))
                meta.update(kind=str(d.get("kind") or ""), problem_text=str(d.get("problem_text") or ""),
                            notes=str(d.get("notes") or ""))
                try:
                    conf = int(float(d.get("confidence")))
                except Exception:
                    conf = None
                add("Gemini vision", d.get("latex") or "", conf)
            except Exception as e:
                errors.append("Gemini: " + str(e)[:140])

        word_problem = meta["kind"] == "word_problem" and meta["problem_text"]
        need_local = (mode in ("compare", "local")) or (not any(c["parses"] for c in cands) and not word_problem)
        if need_local:
            gray = _MxOps.autocontrast(img.convert("L"), cutoff=1).convert("RGB")
            for name, fn in (("Pix2Text", self._read_p2t), ("GOT-OCR 2.0", self._read_got),
                             ("pix2tex (LaTeX-OCR)", self._read_pix2tex)):
                if mode == "auto" and any(c["parses"] for c in cands):
                    break
                try:
                    with self._lock:
                        add(name, fn(gray))
                except ImportError as e:
                    errors.append(f"{name} is not installed ({str(e)[:80]}).")
                except Exception as e:
                    errors.append(f"{name}: {str(e)[:140]}")

        chosen = None
        ok = [c for c in cands if c["parses"]]
        if ok:
            def score(c):
                return 1 + sum(1 for o in ok if o is not c and mx_same(o["latex"], c["latex"]))
            chosen = max(ok, key=lambda c: (score(c), -ok.index(c)))
        elif cands:
            chosen = cands[0]
        agree = [c["engine"] for c in cands if chosen and c is not chosen and mx_same(c["latex"], chosen["latex"])]
        return {"candidates": cands, "chosen": chosen, "agree": agree, "errors": errors,
                "kind": meta["kind"], "problem_text": meta["problem_text"], "notes": meta["notes"]}


def _mx_load_image(data, filename=""):
    im = _MxOps.exif_transpose(_MxImage.open(_mx_io.BytesIO(data)))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = _MxImage.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    if max(im.size) > 2000:
        im.thumbnail((2000, 2000), _MxImage.LANCZOS)
    return _MxOps.expand(im, border=24, fill=(255, 255, 255))


def _mx_opts(src):
    g = (lambda k: (src.get(k) or "").strip()) if src else (lambda k: "")
    return dict(op=g("operation") or "auto", var=g("variable"), lo=g("lo"), hi=g("hi"), point=g("point"), order=g("order"))


def _mx_read_job(ui, data, filename, mode, solve, opts):
    img = _mx_load_image(data, filename)
    rd = ui.math_reader.read(img, mode)
    if not rd["chosen"]:
        why = " ".join(rd["errors"]) or "No maths found in the image."
        raise ValueError("I could not read any maths from this image. " + why)
    out = {"success": True, "read": rd}
    bot = ui.chatbot
    if solve:
        if rd["kind"] == "word_problem" and rd["problem_text"]:
            text = bot.math_text("Solve this problem step by step. Reply in the language of the problem. "
                                 "Write maths in LaTeX between single dollar signs.\n\n" + rd["problem_text"])
            out["solution"] = {"mode": "llm", "markdown": text, "problem": rd["problem_text"]}
        else:
            try:
                out["solution"] = mx_solve(rd["chosen"]["latex"], **opts)
            except Exception as e:
                out["solve_error"] = str(e)
    return out


def _mx_add_routes(self):
    @self.app.route("/api/math_solve", methods=["POST"])
    def math_solve():
        d = request.get_json(silent=True) or {}
        try:
            return jsonify(mx_solve(d.get("expression", ""), **_mx_opts(d)))
        except (ValueError, TimeoutError) as e:
            return jsonify({"success": False, "error": str(e)}), 400
        except Exception as e:
            print("Math error:", e)
            return jsonify({"success": False, "error": "I could not solve that: " + str(e)[:160]}), 400

    @self.app.route("/api/math_read_start", methods=["POST"])
    def math_read_start():
        f = request.files.get("image")
        if not f:
            return jsonify({"success": False, "error": "No image received."}), 400
        data, name = f.read(), f.filename or ""
        mode = request.form.get("mode", "auto")
        solve = request.form.get("solve", "1") == "1"
        opts = _mx_opts(request.form)
        return _job_start(lambda: _mx_read_job(self, data, name, mode, solve, opts))

    @self.app.route("/api/math_explain_start", methods=["POST"])
    def math_explain_start():
        d = request.get_json(silent=True) or {}
        prompt = (MATH_EXPLAIN_PROMPT.replace("__PROBLEM__", str(d.get("problem", ""))[:1500])
                  .replace("__ANSWER__", str(d.get("answer", ""))[:800]))
        return _job_start(lambda: {"success": True, "text": self.chatbot.math_text(prompt)})


def _bot_math_text(self, prompt, temperature=0.2, max_tokens=4096):
    if not self._ensure_client():
        raise RuntimeError(self.last_error or "Gemini is not available.")
    models = [self.chat_model] if self.chat_model else list(self.CHAT_MODELS)
    last = None
    for m in models:
        try:
            resp = self.client.models.generate_content(
                model=m, contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens))
            t = (getattr(resp, "text", None) or "").strip()
            if t:
                self.chat_model = m
                return t
            last = f"{m}: empty answer"
        except Exception as e:
            last = f"{m}: {e}"
            self.last_error = last
    raise RuntimeError(last or "Gemini did not answer.")


SimpleChatbot.math_text = _bot_math_text


# ---- the old calculator methods now use the new engine (and no longer use eval) ----
def _legacy(op):
    def fn(self, expression, variable="x"):
        try:
            r = mx_solve(expression, op, var=variable if op in ("derivative", "integral") else "")
            self.history.append({"expression": expression, "result": r["answer_text"]})
            return {"success": True, "result": r["answer_text"], "solutions": [r["answer_text"]] if op == "solve" else None,
                    "method": r["operation"]}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return fn


AdvancedCalculator.calculate = lambda self, expression: _legacy("auto")(self, expression)
AdvancedCalculator.solve_equation = lambda self, equation: _legacy("solve")(self, equation)
AdvancedCalculator.derivative = lambda self, expression, variable="x": _legacy("derivative")(self, expression, variable)
AdvancedCalculator.integral = lambda self, expression, variable="x": _legacy("integral")(self, expression, variable)

_mx_prev_init = WebUI.__init__


def _mx_init(self):
    _mx_prev_init(self)
    self.math_reader = MathReader(self.chatbot)
    if FLASK_AVAILABLE:
        _mx_add_routes(self)


WebUI.__init__ = _mx_init

# ---------------------------------------------------------------- page
MX_HEAD = ('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">'
           '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>')

MX_PANEL = r"""<!-- CALCULATOR -->
        <div id="calculator" class="tab-content">
            <div class="tab-inner">
                <h2>Math Studio</h2>
                <div class="mx-modes">
                    <button type="button" class="mx-mode active" data-pane="type">⌨️ Type</button>
                    <button type="button" class="mx-mode" data-pane="draw">✍️ Draw</button>
                    <button type="button" class="mx-mode" data-pane="image">📷 Image</button>
                </div>

                <div class="mx-pane active" id="mx-pane-type">
                    <div class="field-row">
                        <div class="input-group">
                            <label>Equation or problem (plain text or LaTeX)</label>
                            <input type="text" id="calc-input" autocomplete="off" spellcheck="false"
                                   placeholder="e.g. x^2-5x+6=0   |   2x+3y=7, x-y=1   |   d/dx (x^3 sin x)   |   ∫_0^1 x^2 dx">
                        </div>
                        <div class="input-group">
                            <label>Variable</label>
                            <input type="text" id="calc-variable" autocomplete="off" placeholder="auto">
                        </div>
                    </div>
                    <div class="mx-keys" id="mx-keys">
                        <button type="button" data-i="^2">x²</button><button type="button" data-i="^">xⁿ</button>
                        <button type="button" data-i="sqrt(">√</button><button type="button" data-i="/">÷</button>
                        <button type="button" data-i="pi">π</button><button type="button" data-i="e^">eˣ</button>
                        <button type="button" data-i="|">|x|</button><button type="button" data-i="(">(</button>
                        <button type="button" data-i=")">)</button><button type="button" data-i="sin(">sin</button>
                        <button type="button" data-i="cos(">cos</button><button type="button" data-i="tan(">tan</button>
                        <button type="button" data-i="ln(">ln</button><button type="button" data-i="log10(">log</button>
                        <button type="button" data-i="∫_0^1  dx">∫</button><button type="button" data-i="lim x->0 ">lim</button>
                        <button type="button" data-i="d/dx ">d/dx</button><button type="button" data-i="≤">≤</button>
                        <button type="button" data-i="≥">≥</button><button type="button" data-i=", ">,</button>
                    </div>
                    <div class="input-group">
                        <label>What to do</label>
                        <select id="mx-op">
                            <option value="auto">Auto-detect (recommended)</option>
                            <option value="solve">Solve equation</option>
                            <option value="simplify">Simplify / calculate</option>
                            <option value="expand">Expand</option>
                            <option value="factor">Factor</option>
                            <option value="apart">Partial fractions</option>
                            <option value="derivative">Derivative</option>
                            <option value="integral">Integral (indefinite or definite)</option>
                            <option value="limit">Limit</option>
                            <option value="sum">Sum (Σ)</option>
                            <option value="series">Taylor series</option>
                            <option value="plot">Graph</option>
                        </select>
                    </div>
                    <details class="mx-more">
                        <summary>More options (bounds, limit point, order)</summary>
                        <div class="mx-opts">
                            <div><label>From / lower bound</label><input type="text" id="mx-lo" placeholder="0"></div>
                            <div><label>To / upper bound</label><input type="text" id="mx-hi" placeholder="pi or oo"></div>
                            <div><label>At point / limit point</label><input type="text" id="mx-point" placeholder="0"></div>
                            <div><label>Order / direction (+ or -)</label><input type="text" id="mx-order" placeholder="2"></div>
                        </div>
                    </details>
                    <div class="button-group">
                        <button class="generate-button" id="mx-go" onclick="calculate()"><span>Solve</span></button>
                    </div>
                    <div class="mx-ex-title">Try an example</div>
                    <div class="mx-examples" id="mx-examples"></div>
                </div>

                <div class="mx-pane" id="mx-pane-draw">
                    <div class="mx-tools">
                        <button type="button" class="mx-tool on" data-tool="pen">✏️ Pen</button>
                        <button type="button" class="mx-tool" data-tool="eraser">🧽 Eraser</button>
                        <label class="mx-size">Size <input type="range" id="mx-size" min="2" max="14" value="5"></label>
                        <button type="button" class="mx-tool" id="mx-undo">↩ Undo</button>
                        <button type="button" class="mx-tool" id="mx-clear">🗑 Clear</button>
                    </div>
                    <div class="mx-canvas-wrap">
                        <canvas id="mx-canvas" width="1200" height="380"></canvas>
                        <div class="mx-ph" id="mx-ph">Write your equation here with the mouse, a pen, or your finger</div>
                    </div>
                    <div class="mx-actions">
                        <button class="generate-button" id="mx-read-draw"><span>Read &amp; solve</span></button>
                        <select class="mx-engine" title="How the handwriting is read">
                            <option value="auto">Reader: best (auto)</option>
                            <option value="compare">Reader: compare all models</option>
                            <option value="local">Reader: open-source models only</option>
                        </select>
                        <label class="mx-chk"><input type="checkbox" id="mx-auto"> read automatically when I pause</label>
                    </div>
                </div>

                <div class="mx-pane" id="mx-pane-image">
                    <div class="mx-drop" id="mx-drop">
                        <div class="mx-drop-t">📷 Drop a photo of the equation here, paste it (Ctrl+V), or choose a file</div>
                        <div class="oc-row" style="justify-content:center">
                            <button type="button" class="oc-btn" id="mx-choose">Choose image</button>
                            <button type="button" class="oc-btn" id="mx-camera">Take a photo</button>
                        </div>
                        <input type="file" id="mx-file" accept="image/*" hidden>
                        <input type="file" id="mx-cam" accept="image/*" capture="environment" hidden>
                        <img id="mx-preview" alt="" style="display:none">
                    </div>
                    <div class="mx-actions">
                        <button class="generate-button" id="mx-read-img"><span>Read &amp; solve</span></button>
                        <select class="mx-engine" title="How the image is read">
                            <option value="auto">Reader: best (auto)</option>
                            <option value="compare">Reader: compare all models</option>
                            <option value="local">Reader: open-source models only</option>
                        </select>
                    </div>
                </div>

                <div class="output" id="calc-output"></div>
            </div>
        </div>

        """

MX_CSS = r"""
#calc-output{white-space:normal}
.mx-modes{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}
.mx-mode{padding:10px 20px;border-radius:999px;cursor:pointer;font-size:1em;font-weight:700;color:#ffaa5c;
  background:rgba(0,0,0,.55);border:2px solid rgba(221,132,72,.5);transition:all .2s}
.mx-mode:hover{background:rgba(221,132,72,.3);color:#fff}
.mx-mode.active{background:linear-gradient(135deg,#dd8448,#ffaa5c);color:#fff;border-color:#ffaa5c}
.mx-pane{display:none}.mx-pane.active{display:block}
.mx-keys{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 16px}
.mx-keys button,.mx-tool{padding:7px 12px;min-width:42px;border-radius:10px;cursor:pointer;font-size:.95em;color:#ffaa5c;
  background:rgba(0,0,0,.55);border:1px solid rgba(221,132,72,.6)}
.mx-keys button:hover,.mx-tool:hover{background:rgba(221,132,72,.3);color:#fff}
.mx-tool.on{background:rgba(221,132,72,.55);color:#fff}
.mx-more{margin:0 0 16px}.mx-more summary{cursor:pointer;color:#ffaa5c;font-weight:600}
.mx-opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:10px}
.mx-opts label{font-size:.9em}
.mx-ex-title{margin:16px 0 6px;font-size:.9em;opacity:.8}
.mx-examples{display:flex;flex-wrap:wrap;gap:8px}
.mx-examples button{padding:6px 12px;border-radius:999px;cursor:pointer;font-size:.88em;color:#e0e0e0;background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.25);font-family:'Courier New',monospace}
.mx-examples button:hover{background:rgba(221,132,72,.3);color:#fff}
.mx-tools{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.mx-size{display:flex;align-items:center;gap:6px;font-size:.9em}.mx-size input{width:110px;padding:0;border:0;background:none}
.mx-canvas-wrap{position:relative;border-radius:16px;overflow:hidden;border:2px solid rgba(221,132,72,.6);background:#fff;line-height:0}
#mx-canvas{width:100%;height:auto;aspect-ratio:1200/380;touch-action:none;cursor:crosshair;display:block;background:#fff}
.mx-ph{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:20px;line-height:1.4;
  color:#9aa3b5;font-size:1.1em;pointer-events:none}
.mx-actions{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:14px}
.mx-engine{width:auto!important;max-width:260px;padding:9px 12px!important;font-size:14px!important;border-radius:12px!important}
.mx-chk{display:flex;align-items:center;gap:6px;font-size:.9em;font-weight:400}.mx-chk input{width:auto}
.mx-drop{border:2px dashed rgba(221,132,72,.7);border-radius:16px;padding:22px;text-align:center;background:rgba(0,0,0,.25);transition:all .2s}
.mx-drop.over{background:rgba(221,132,72,.25);border-color:#ffaa5c}
.mx-drop-t{margin-bottom:12px}
#mx-preview{max-width:100%;max-height:260px;border-radius:12px;margin:14px auto 0;background:#fff}
.mx-in{padding:6px 0;overflow-x:auto}
.mx-ans{margin:12px 0;padding:14px 18px;border-radius:16px;background:rgba(60,200,120,.12);border:2px solid rgba(60,200,120,.6);overflow-x:auto}
.mx-ans-l{font-size:.8em;opacity:.8;text-transform:uppercase;letter-spacing:.08em}
.mx-ans .katex-display{margin:.5em 0}
.mx-ans-t{font-family:'Courier New',monospace;opacity:.85;font-size:.95em;word-break:break-word}
.mx-extra{margin:8px 0;padding:8px 12px;border-radius:12px;background:rgba(255,255,255,.07);overflow-x:auto}
.mx-extra b{color:#ffaa5c}
.mx-steps{margin:0;padding-left:22px}.mx-steps li{margin:8px 0}
.mx-steps .katex-display{margin:.3em 0;text-align:left}
.mx-plot{width:100%;max-width:760px;height:auto;border-radius:12px;background:#fff;display:block;margin:8px 0}
.mx-read{margin-bottom:16px;padding:14px;border-radius:16px;background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.2)}
.mx-rendered{min-height:34px;padding:6px 0;overflow-x:auto}
.mx-rec{width:100%;font-family:'Courier New',monospace;margin-top:6px}
.mx-cands{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.mx-cand{padding:6px 12px;border-radius:12px;cursor:pointer;font-size:.88em;color:inherit;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.3);text-align:left}
.mx-cand:hover{background:rgba(221,132,72,.3)}
.mx-cand.bad{opacity:.6}
.mx-md{line-height:1.7}
body.light .mx-mode,body.light .mx-keys button,body.light .mx-tool{color:#b8325a;background:rgba(255,255,255,.7);border-color:#ff7aa2}
body.light .mx-mode.active,body.light .mx-tool.on{color:#fff;background:linear-gradient(135deg,#ff7aa2,#ffb347)}
body.light .mx-more summary,body.light .mx-extra b{color:#b8325a}
body.light .mx-examples button{color:#12304f;background:rgba(255,255,255,.6);border-color:rgba(60,120,200,.4)}
body.light .mx-read,body.light .mx-extra{background:rgba(255,255,255,.5);border-color:rgba(60,120,200,.3)}
body.light .mx-drop{background:rgba(255,255,255,.4)}
@media (max-width:700px){.mx-mode{padding:9px 14px;font-size:.92em}.mx-keys button{min-width:38px;padding:6px 9px}
  .mx-actions .generate-button{flex:1 1 100%}.mx-engine{max-width:100%;flex:1 1 100%}}
"""

MX_JS = r"""
(function () {
  const $ = id => document.getElementById(id);
  const esc = s => escapeHtml(String(s == null ? '' : s));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const tex = (s, display) => {
    try { if (window.katex) return katex.renderToString(String(s), { displayMode: !!display, throwOnError: false }); } catch (e) {}
    return '<code>' + esc(s) + '</code>';
  };
  const out = $('calc-output');

  // ---------------- modes ----------------
  const modes = document.querySelectorAll('.mx-mode');
  function setMode(name) {
    modes.forEach(b => b.classList.toggle('active', b.dataset.pane === name));
    document.querySelectorAll('.mx-pane').forEach(p => p.classList.toggle('active', p.id === 'mx-pane-' + name));
  }
  modes.forEach(b => b.addEventListener('click', () => setMode(b.dataset.pane)));

  // ---------------- keypad + examples ----------------
  const inp = $('calc-input');
  $('mx-keys').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    const t = b.dataset.i, s = inp.selectionStart == null ? inp.value.length : inp.selectionStart, en = inp.selectionEnd == null ? s : inp.selectionEnd;
    inp.value = inp.value.slice(0, s) + t + inp.value.slice(en);
    inp.focus(); inp.setSelectionRange(s + t.length, s + t.length);
  });
  ['x^2-5x+6=0', '2x+3y=7, x-y=1', 'x^2+y^2=25, y=x+1', 'sin(x)=1/2', 'e^(2x)-3e^x+2=0', 'x^2-4<0',
   'd/dx (x^3 sin x)', '∫_0^pi sin(x) dx', '∫ x e^x dx', 'lim x->0 sin(x)/x', "y'' + y = 0", 'dy/dx = 2y',
   '[[1,2],[3,4]]', '(x^2-1)/(x-1)', 'factor x^3-2x^2-x+2', 'sum n^2 from 1 to 10'].forEach(t => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = t;
    b.addEventListener('click', () => { inp.value = t; $('mx-op').value = 'auto'; solveText(t.replace(/^sum (.*) from 1 to 10$/, '\\sum_{n=1}^{10} $1')); });
    $('mx-examples').appendChild(b);
  });

  // ---------------- output skeleton ----------------
  function skeleton() {
    out.innerHTML = '<div id="mx-readbox"></div><div id="mx-sol"></div>';
  }
  function loader(stage) {
    out.innerHTML = '<div class="an-stage"><span id="mx-stage">' + esc(stage) + '</span><span class="an-dots"><i></i><i></i><i></i></span></div>' +
                    '<div class="an-hint" id="mx-hint"></div>';
    const t0 = Date.now();
    const iv = setInterval(() => {
      const h = $('mx-hint'); if (!h) { clearInterval(iv); return; }
      const s = Math.floor((Date.now() - t0) / 1000);
      if (s > 12) h.textContent = 'Still working (' + s + 's). The first time a model is used it has to download, so this can take a few minutes.';
    }, 1000);
    return () => clearInterval(iv);
  }
  function opts() {
    return { operation: $('mx-op').value, variable: $('calc-variable').value.trim(), lo: $('mx-lo').value.trim(),
             hi: $('mx-hi').value.trim(), point: $('mx-point').value.trim(), order: $('mx-order').value.trim() };
  }
  async function pollJob(url, init) {
    const r = await fetch(url, init);
    const t = await r.text();
    let s; try { s = JSON.parse(t); } catch (e) { throw new Error('The server answered with a web page (HTTP ' + r.status + '). Reload and try again.'); }
    if (!s.job) throw new Error(s.error || 'Could not start.');
    let fails = 0;
    while (true) {
      await sleep(800);
      try {
        const p = JSON.parse(await (await fetch('/api/job/' + s.job)).text());
        fails = 0;
        if (p.done) return p.result;
      } catch (e) { if (++fails > 120) throw new Error('Lost connection to the server.'); }
    }
  }

  // ---------------- markdown with maths (AI explanations) ----------------
  function mdMath(text) {
    const parts = [];
    const t = String(text).replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (m, a, b) => { parts.push(tex(a || b, !!a)); return '@@M' + (parts.length - 1) + '@@'; });
    return renderMarkdown(t).replace(/@@M(\d+)@@/g, (m, i) => parts[+i]);
  }

  // ---------------- graph ----------------
  function drawPlot(cv, pl) {
    const g = cv.getContext('2d'), W = cv.width, H = cv.height, m = { l: 46, r: 14, t: 26, b: 28 };
    g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
    const ys = [];
    pl.series.forEach(s => s.pts.forEach(p => { if (p[1] !== null) ys.push(p[1]); }));
    if (!ys.length) { g.fillStyle = '#555'; g.fillText('Nothing to draw for real numbers.', 20, 30); return; }
    ys.sort((a, b) => a - b);
    let lo = ys[Math.floor(ys.length * 0.03)], hi = ys[Math.floor(ys.length * 0.97)];
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
    const X = x => m.l + (x - pl.xmin) / (pl.xmax - pl.xmin) * (W - m.l - m.r);
    const Y = y => H - m.b - (y - lo) / (hi - lo) * (H - m.t - m.b);
    g.font = '12px Segoe UI, Arial'; g.lineWidth = 1;
    const nice = (a, b) => { const r = (b - a) / 6, p = Math.pow(10, Math.floor(Math.log10(r))), n = r / p; return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p; };
    const sx = nice(pl.xmin, pl.xmax), sy = nice(lo, hi);
    g.fillStyle = '#667'; g.strokeStyle = '#e3e6ee';
    for (let x = Math.ceil(pl.xmin / sx) * sx; x <= pl.xmax; x += sx) { g.beginPath(); g.moveTo(X(x), m.t); g.lineTo(X(x), H - m.b); g.stroke(); g.fillText((+x.toPrecision(4)).toString(), X(x) - 8, H - 10); }
    for (let y = Math.ceil(lo / sy) * sy; y <= hi; y += sy) { g.beginPath(); g.moveTo(m.l, Y(y)); g.lineTo(W - m.r, Y(y)); g.stroke(); g.fillText((+y.toPrecision(4)).toString(), 4, Y(y) + 4); }
    g.strokeStyle = '#333'; g.lineWidth = 1.5;
    if (lo < 0 && hi > 0) { g.beginPath(); g.moveTo(m.l, Y(0)); g.lineTo(W - m.r, Y(0)); g.stroke(); }
    if (pl.xmin < 0 && pl.xmax > 0) { g.beginPath(); g.moveTo(X(0), m.t); g.lineTo(X(0), H - m.b); g.stroke(); }
    const cols = ['#e8590c', '#1c7ed6', '#2f9e44'];
    pl.series.forEach((s, i) => {
      g.strokeStyle = cols[i % 3]; g.lineWidth = 2.4; g.beginPath();
      let pen = false;
      s.pts.forEach(p => {
        if (p[1] === null || p[1] < lo - (hi - lo) * 2 || p[1] > hi + (hi - lo) * 2) { pen = false; return; }
        if (!pen) { g.moveTo(X(p[0]), Y(p[1])); pen = true; } else g.lineTo(X(p[0]), Y(p[1]));
      });
      g.stroke();
      g.fillStyle = cols[i % 3]; g.fillText(s.label.slice(0, 30), m.l + 8 + i * 230, 16);
    });
    g.fillStyle = '#d6336c';
    (pl.marks || []).forEach(x => { if (x >= pl.xmin && x <= pl.xmax && lo < 0 && hi > 0) { g.beginPath(); g.arc(X(x), Y(0), 5, 0, 7); g.fill(); } });
  }

  // ---------------- render a solution ----------------
  function renderSolution(d) {
    if (d.mode === 'llm' || d.markdown) {
      return '<div class="cr-status ok">🤖 Word problem solved by the AI</div><div class="mx-md">' + mdMath(d.markdown) + '</div>' +
             '<div class="ppl-note">This answer comes from the AI language model, not from the exact maths engine. Check it.</div>';
    }
    let h = '<div class="cr-status ok">🧮 ' + esc(d.operation) + '</div>';
    h += '<div class="mx-in">' + tex(d.input_latex, true) + '</div>';
    h += '<div class="mx-ans"><div class="mx-ans-l">Answer</div>' + tex(d.answer_latex, true) + '<div class="mx-ans-t">' + esc(d.answer_text) + '</div></div>';
    (d.extra || []).forEach(x => { h += '<div class="mx-extra"><b>' + esc(x.label) + ':</b> ' + tex(x.latex, false) + '</div>'; });
    if (d.steps && d.steps.length) {
      h += '<div class="cr-title">🪜 Steps</div><ol class="mx-steps">' + d.steps.map(s =>
        '<li>' + esc(s.text) + (s.latex ? '<div>' + tex(s.latex, true) + '</div>' : '') + '</li>').join('') + '</ol>';
    }
    if (d.plot) h += '<div class="cr-title">📈 Graph</div><canvas class="mx-plot" width="760" height="340"></canvas>';
    if (d.notes && d.notes.length) h += '<div class="cr-title">📝 Notes</div><ul>' + d.notes.map(n => '<li>' + esc(n) + '</li>').join('') + '</ul>';
    h += '<div class="oc-row"><button type="button" class="oc-btn" id="mx-explain">🤖 Explain step by step with AI</button></div><div id="mx-explain-out" class="mx-md"></div>';
    return h;
  }
  function showSolution(d, problemText) {
    const box = $('mx-sol'); box.innerHTML = renderSolution(d);
    const pc = box.querySelector('canvas.mx-plot'); if (pc && d.plot) drawPlot(pc, d.plot);
    const eb = $('mx-explain');
    if (eb) eb.addEventListener('click', async () => {
      const o = $('mx-explain-out'); eb.disabled = true; o.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
      try {
        const r = await pollJob('/api/math_explain_start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ problem: problemText, answer: d.answer_text }) });
        o.innerHTML = r.success ? mdMath(r.text) : esc(r.error || 'Could not explain.');
      } catch (e) { o.textContent = 'Error: ' + e.message; }
      eb.disabled = false;
    });
  }
  function showError(msg) {
    $('mx-sol').innerHTML = '<div class="cr-status warn">⚠️ ' + esc(msg) + '</div>' +
      '<div class="ppl-note">If the text above was read wrongly, correct it and press Solve again.</div>';
  }

  // ---------------- solve typed / corrected text ----------------
  async function solveText(text) {
    text = (text || '').trim();
    if (!text) { out.textContent = 'Please enter an equation or problem.'; return; }
    if (!$('mx-sol')) skeleton();
    $('mx-sol').innerHTML = '<div class="an-stage"><span>Solving</span><span class="an-dots"><i></i><i></i><i></i></span></div>';
    try {
      const body = Object.assign({ expression: text }, opts());
      const r = await fetch('/api/math_solve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const t = await r.text(); let d;
      try { d = JSON.parse(t); } catch (e) { throw new Error('The server answered with a web page (HTTP ' + r.status + ').'); }
      if (!d.success) { showError(d.error || 'Could not solve this.'); return; }
      showSolution(d, text);
      out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) { showError(e.message); }
  }
  window.calculate = function () { skeleton(); return solveText(inp.value); };
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); window.calculate(); } });

  // ---------------- show what was read ----------------
  function renderRead(rd) {
    const c = rd.chosen;
    let h = '<div class="mx-read"><div class="cr-title" style="margin-top:0">👁️ What I read <small style="font-weight:400;opacity:.8">(fix it if something is wrong, then press Solve)</small></div>';
    h += '<div class="mx-rendered" id="mx-rendered">' + tex(c.latex, true) + '</div>';
    h += '<input type="text" class="mx-rec" id="mx-rec" value="' + esc(c.latex) + '" spellcheck="false">';
    h += '<div class="oc-row"><button type="button" class="oc-btn" id="mx-resolve">▶ Solve this</button></div>';
    const sure = [];
    if (rd.agree && rd.agree.length) sure.push('✅ ' + (rd.agree.length + 1) + ' models read the same thing');
    if (c.confidence != null) sure.push('confidence ' + c.confidence + '%');
    if (!c.parses) sure.push('⚠️ the maths engine cannot parse this yet, please correct it');
    if (sure.length) h += '<p><small>' + esc(sure.join(' · ')) + '</small></p>';
    if (rd.notes) h += '<p><small>📝 ' + esc(rd.notes) + '</small></p>';
    if (rd.candidates.length > 1) {
      h += '<div class="cr-title">🔀 Other readings</div><div class="mx-cands">' + rd.candidates.map((x, i) =>
        '<button type="button" class="mx-cand' + (x.parses ? '' : ' bad') + '" data-i="' + i + '"><b>' + esc(x.engine) + '</b><br>' + tex(x.latex, false) + '</button>').join('') + '</div>';
    }
    if (rd.errors && rd.errors.length) h += '<details><summary><small>Engines that did not work (' + rd.errors.length + ')</small></summary><ul>' + rd.errors.map(e => '<li><small>' + esc(e) + '</small></li>').join('') + '</ul></details>';
    h += '</div>';
    return h;
  }
  function wireRead(rd) {
    const rec = $('mx-rec');
    const upd = () => { $('mx-rendered').innerHTML = tex(rec.value, true); };
    rec.addEventListener('input', upd);
    rec.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); solveText(rec.value); } });
    $('mx-resolve').addEventListener('click', () => solveText(rec.value));
    document.querySelectorAll('.mx-cand').forEach(b => b.addEventListener('click', () => {
      rec.value = rd.candidates[+b.dataset.i].latex; upd(); solveText(rec.value);
    }));
  }

  // ---------------- read an image (drawing or photo) ----------------
  let busy = false;
  async function readBlob(blob, engineSel) {
    if (busy) return; busy = true;
    const stop = loader('Reading the maths');
    try {
      const fd = new FormData();
      fd.append('image', blob, 'math.png');
      fd.append('mode', engineSel ? engineSel.value : 'auto');
      fd.append('solve', '1');
      const o = opts(); Object.keys(o).forEach(k => fd.append(k, o[k]));
      const res = await pollJob('/api/math_read_start', { method: 'POST', body: fd });
      stop();
      if (!res.success) { out.innerHTML = '<div class="cr-status warn">⚠️ ' + esc(res.error || 'Could not read the image.') + '</div>'; return; }
      skeleton();
      $('mx-readbox').innerHTML = renderRead(res.read);
      wireRead(res.read);
      const rec = res.read.chosen.latex;
      if (res.solution) showSolution(res.solution, res.read.problem_text || rec);
      else if (res.solve_error) showError(res.solve_error);
      out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (e) {
      stop(); out.innerHTML = '<div class="cr-status warn">⚠️ ' + esc(e.message) + '</div>';
    } finally { busy = false; }
  }

  // ---------------- drawing surface (mouse, pen, finger) ----------------
  (function () {
    const cv = $('mx-canvas'), ctx = cv.getContext('2d'), ph = $('mx-ph');
    let strokes = [], hist = [], cur = null, tool = 'pen', penSeen = false, timer = null;
    const sizeEl = $('mx-size');
    const K = () => cv.width / cv.getBoundingClientRect().width;           // CSS px -> canvas px
    const pt = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height, e.pressure || 0.5]; };
    function line(s, a, b) {
      ctx.strokeStyle = '#111'; ctx.lineCap = ctx.lineJoin = 'round';
      ctx.lineWidth = s.w * (0.7 + (b[2] || 0.5) * 0.6);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    function drawStroke(s) {
      if (s.p.length === 1) { ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(s.p[0][0], s.p[0][1], s.w / 2, 0, 7); ctx.fill(); return; }
      for (let i = 1; i < s.p.length; i++) line(s, s.p[i - 1], s.p[i]);
    }
    function redraw() {
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
      strokes.forEach(drawStroke);
      ph.style.display = strokes.length || cur ? 'none' : 'grid';
    }
    function snapshot() { hist.push(strokes.slice()); if (hist.length > 60) hist.shift(); }
    function eraseAt(p) {
      const r = (+sizeEl.value) * 4 * K() / 1.3 + 6;
      const keep = strokes.filter(s => !s.p.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < r));
      if (keep.length !== strokes.length) { strokes = keep; redraw(); }
    }
    cv.addEventListener('pointerdown', e => {
      if (e.pointerType === 'pen') penSeen = true;
      if (e.pointerType === 'touch' && penSeen) return;                    // palm rejection
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      e.preventDefault(); cv.setPointerCapture(e.pointerId);
      clearTimeout(timer); snapshot();
      const p = pt(e);
      if (tool === 'eraser') { cur = { erase: true }; eraseAt(p); return; }
      cur = { w: +sizeEl.value * K() * 0.9, p: [p] };
      ph.style.display = 'none';
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(p[0], p[1], cur.w / 2, 0, 7); ctx.fill();
    });
    cv.addEventListener('pointermove', e => {
      if (!cur) return;
      e.preventDefault();
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      (evs.length ? evs : [e]).forEach(ev => {
        const p = pt(ev);
        if (cur.erase) { eraseAt(p); return; }
        const last = cur.p[cur.p.length - 1];
        if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 1.2) return;
        line(cur, last, p); cur.p.push(p);
      });
    });
    function up() {
      if (!cur) return;
      const c = cur; cur = null;
      if (!c.erase) strokes.push(c);
      redraw();
      if ($('mx-auto').checked && strokes.length) { clearTimeout(timer); timer = setTimeout(readDrawing, 1800); }
    }
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('contextmenu', e => e.preventDefault());

    document.querySelectorAll('.mx-tool[data-tool]').forEach(b => b.addEventListener('click', () => {
      tool = b.dataset.tool;
      document.querySelectorAll('.mx-tool[data-tool]').forEach(x => x.classList.toggle('on', x === b));
      cv.style.cursor = tool === 'eraser' ? 'cell' : 'crosshair';
    }));
    $('mx-undo').addEventListener('click', () => { if (hist.length) { strokes = hist.pop(); redraw(); } });
    $('mx-clear').addEventListener('click', () => { if (strokes.length) snapshot(); strokes = []; redraw(); });

    function exportPng() {
      if (!strokes.length) return Promise.resolve(null);
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      strokes.forEach(s => s.p.forEach(q => { x0 = Math.min(x0, q[0]); y0 = Math.min(y0, q[1]); x1 = Math.max(x1, q[0]); y1 = Math.max(y1, q[1]); }));
      const pad = 50, w = Math.ceil(x1 - x0 + pad * 2), h = Math.ceil(y1 - y0 + pad * 2);
      const c = document.createElement('canvas'); c.width = w; c.height = h;
      const g = c.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h);
      g.translate(pad - x0, pad - y0);
      g.strokeStyle = '#111'; g.fillStyle = '#111'; g.lineCap = g.lineJoin = 'round';
      strokes.forEach(s => {
        if (s.p.length === 1) { g.beginPath(); g.arc(s.p[0][0], s.p[0][1], s.w / 2, 0, 7); g.fill(); return; }
        for (let i = 1; i < s.p.length; i++) { g.lineWidth = s.w * (0.7 + (s.p[i][2] || 0.5) * 0.6); g.beginPath(); g.moveTo(s.p[i - 1][0], s.p[i - 1][1]); g.lineTo(s.p[i][0], s.p[i][1]); g.stroke(); }
      });
      return new Promise(res => c.toBlob(res, 'image/png'));
    }
    async function readDrawing() {
      clearTimeout(timer);
      const blob = await exportPng();
      if (!blob) { out.textContent = 'Write something on the board first.'; return; }
      readBlob(blob, document.querySelector('#mx-pane-draw .mx-engine'));
    }
    $('mx-read-draw').addEventListener('click', readDrawing);
    redraw();
  })();

  // ---------------- image upload (file, camera, drag and drop, paste) ----------------
  (function () {
    const drop = $('mx-drop'), prev = $('mx-preview'), file = $('mx-file'), cam = $('mx-cam');
    let blob = null;
    function setImage(f) {
      if (!f || !/^image\//.test(f.type || 'image/')) return;
      blob = f; prev.src = URL.createObjectURL(f); prev.style.display = 'block';
    }
    $('mx-choose').addEventListener('click', () => file.click());
    $('mx-camera').addEventListener('click', () => cam.click());
    file.addEventListener('change', () => setImage(file.files[0]));
    cam.addEventListener('change', () => setImage(cam.files[0]));
    ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', e => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) setImage(f); });
    document.addEventListener('paste', e => {
      if (!$('calculator').classList.contains('active')) return;
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const it of items) if (it.type && it.type.indexOf('image') === 0) { setMode('image'); setImage(it.getAsFile()); e.preventDefault(); break; }
    });
    $('mx-read-img').addEventListener('click', () => {
      if (!blob) { out.textContent = 'Choose or drop an image first.'; return; }
      readBlob(blob, document.querySelector('#mx-pane-image .mx-engine'));
    });
  })();
})();
"""

_mx_prev_get_html = WebUI.get_html


def _mx_get_html(self) -> str:
    html = _mx_prev_get_html(self)
    if "<!-- CALCULATOR -->" in html and "<!-- CODE CHECKER -->" in html:
        html = re.sub(r"<!-- CALCULATOR -->.*?<!-- CODE CHECKER -->",
                      lambda m: MX_PANEL + "<!-- CODE CHECKER -->", html, count=1, flags=re.S)
    else:
        print("⚠️ math patch: could not find the calculator panel markers, panel not replaced")
    html = html.replace("</head>", MX_HEAD + "<style>" + MX_CSS + "</style></head>", 1)
    return html.replace("</body>", "<script>" + MX_JS + "</script></body>", 1)


WebUI.get_html = _mx_get_html

# ============================================================================
# MULTI-USER + SAVED CHATS + SAVED MATH HISTORY PATCH
# ============================================================================
import copy
from flask import g as _flask_g

_CUR = threading.local()                 # which user the current thread works for
SESSIONS = None
PIP_DATA_DIR = os.getenv("PIP_DATA_DIR", "pip_data")
os.makedirs(PIP_DATA_DIR, exist_ok=True)
_SID_RX = re.compile(r"[0-9a-f]{32}")


# ---- 1) background threads started during a request remember the user ----

def _is_pip_patch(fn):
    """True if fn is one of OUR patched thread methods (not a real original)."""
    return getattr(fn, "_pip_patched", False) or \
        getattr(fn, "__name__", "") in ("_thr_init2", "_thr_run2")


# Undo ANY previously applied patch (old `_pip_orig_*` or new `_pip_original_*`
# naming) so re-running this file in the same Colab session can never stack
# patches or recurse.
_saved_candidates = {}
for _attr in ("_pip_original_init", "_pip_original_run",
              "_pip_orig_init", "_pip_orig_run"):
    _saved = getattr(threading.Thread, _attr, None)
    if _saved is not None:
        _saved_candidates[_attr] = _saved
        delattr(threading.Thread, _attr)

# Pick the first candidate that is NOT itself a patched function.
_real_init = next((f for a, f in _saved_candidates.items()
                   if a.endswith("_init") and not _is_pip_patch(f)), None)
_real_run = next((f for a, f in _saved_candidates.items()
                  if a.endswith("_run") and not _is_pip_patch(f)), None)

if _saved_candidates and (_real_init is None or _real_run is None):
    # Saved "originals" exist but they are all patched functions — the kernel's
    # threading.Thread is corrupted. Reload the threading module to get a
    # fresh, unpatched Thread class.
    try:
        import importlib
        threading = importlib.reload(threading)
        print("♻️  Repaired corrupted threading.Thread — continuing normally.")
    except Exception as _re:
        print("⚠️ threading.Thread is corrupted and could not be auto-repaired:", _re)
        print("   Please RESTART the Colab runtime once (Runtime → Restart "
              "session), then run this file again.")

if _real_init is not None:
    threading.Thread.__init__ = _real_init
if _real_run is not None:
    threading.Thread.run = _real_run


# Save the (now original) methods once on the class itself.
threading.Thread._pip_original_init = threading.Thread.__init__
threading.Thread._pip_original_run = threading.Thread.run

_thr_init_orig = threading.Thread._pip_original_init
_thr_run_orig = threading.Thread._pip_original_run


def _thr_init2(self, *a, **k):
    _thr_init_orig(self, *a, **k)
    self._pip_sid = getattr(_CUR, "sid", None)


_thr_init2._pip_patched = True


def _thr_run2(self):
    _CUR.sid = getattr(self, "_pip_sid", None)
    return _thr_run_orig(self)


_thr_run2._pip_patched = True


threading.Thread.__init__ = _thr_init2
threading.Thread.run = _thr_run2

# ---- 2) per-user storage on disk ----
class _Store:
    MAX_CHATS, MAX_MSGS, MAX_MATH = 100, 300, 60

    def __init__(self):
        self.lock = threading.RLock()
        self.cache = {}

    def _path(self, sid):
        return os.path.join(PIP_DATA_DIR, sid + ".json")

    def get(self, sid):
        with self.lock:
            d = self.cache.get(sid)
            if d is None:
                try:
                    with open(self._path(sid), encoding="utf-8") as f:
                        d = json.load(f)
                except Exception:
                    d = {}
                d.setdefault("chats", {})
                d.setdefault("active", None)
                d.setdefault("math", [])
                if len(self.cache) > 300:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[sid] = d
            return d

    def save(self, sid, d):
        with self.lock:
            tmp = self._path(sid) + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False)
                os.replace(tmp, self._path(sid))
            except Exception as e:
                print("Could not save user data:", e)

    @staticmethod
    def hist_of(c):
        if not c:
            return []
        return [{"role": "user" if m["role"] == "user" else "model", "text": m["text"]}
                for m in c["messages"][-40:]]

    def history(self, sid):
        with self.lock:
            d = self.get(sid)
            return self.hist_of(d["chats"].get(d["active"]))

    def log_turn(self, sid, user_text, bot_text):
        with self.lock:
            d = self.get(sid)
            cid = d["active"]
            now = time.time()
            if cid not in d["chats"]:
                cid = uuid.uuid4().hex[:12]
                title = " ".join(user_text.split())[:40] or "New chat"
                d["chats"][cid] = {"id": cid, "title": title, "created": now, "updated": now, "messages": []}
                d["active"] = cid
                if len(d["chats"]) > self.MAX_CHATS:
                    oldest = min(d["chats"].values(), key=lambda c: c["updated"])["id"]
                    if oldest != cid:
                        d["chats"].pop(oldest, None)
            c = d["chats"][cid]
            c["messages"] += [{"role": "user", "text": user_text}, {"role": "assistant", "text": bot_text}]
            del c["messages"][:-self.MAX_MSGS]
            c["updated"] = now
            self.save(sid, d)


STORE = _Store()


# ---- 3) one chatbot per user (heavy parts are shared) ----
class _Sessions:
    MAX = 300

    def __init__(self, base):
        self.base = base
        self.bots = {}
        self.lock = threading.Lock()

    def _make(self, sid):
        base = self.base
        base._ensure_client()
        base._ensure_knowledge()          # embed the built-in facts once, not once per user
        b = copy.copy(base)               # shares client, knowledge, text corrector
        b.history = STORE.history(sid)    # restore the active saved chat
        b.memory = []
        b.image_ctx = None
        b._lock = threading.Lock()
        b.sid = sid
        return b

    def bot(self):
        sid = getattr(_CUR, "sid", None)
        if not sid:
            return self.base
        with self.lock:
            b = self.bots.get(sid)
            if b is None:
                b = self._make(sid)
                self.bots[sid] = b
                if len(self.bots) > self.MAX:
                    others = [k for k in self.bots if k != sid]
                    self.bots.pop(min(others, key=lambda k: self.bots[k]._seen), None)
            b._seen = time.time()
            return b


class _BotProxy:
    """Looks like one chatbot, but every call goes to the bot of the current user."""
    def __getattr__(self, name):
        return getattr(SESSIONS.bot(), name)

    def __setattr__(self, name, value):
        setattr(SESSIONS.bot(), name, value)


# save every successful chat turn into the user's active chat
_mu_prev_chat = SimpleChatbot.chat


def _mu_chat(self, message):
    n = len(self.history)
    reply = _mu_prev_chat(self, message)
    sid = getattr(self, "sid", None)
    if sid and len(self.history) > n:
        STORE.log_turn(sid, message, reply)
    return reply


SimpleChatbot.chat = _mu_chat

# the shared Hugging Face fake-news model answers one request at a time
_FN_LOCK = threading.Lock()
_fn_ml_unlocked = _fn_ml


def _fn_ml(self, text):
    with _FN_LOCK:
        return _fn_ml_unlocked(self, text)


def _apply_chat(bot, c):
    bot.history = STORE.hist_of(c)
    bot.memory = []
    bot.image_ctx = None


# ---- 4) routes ----
def _mu_add_routes(self):
    app = self.app

    @app.before_request
    def _sid_in():
        sid = request.cookies.get("pip_sid", "")
        _flask_g.new_sid = False
        if not _SID_RX.fullmatch(sid):
            sid = uuid.uuid4().hex
            _flask_g.new_sid = True
        _flask_g.sid = sid
        _CUR.sid = sid

    @app.after_request
    def _sid_out(resp):
        if getattr(_flask_g, "new_sid", False):
            resp.set_cookie("pip_sid", _flask_g.sid, max_age=31536000, httponly=True, samesite="Lax")
        return resp

    @app.teardown_request
    def _sid_clear(exc):
        _CUR.sid = None

    # ----- chats -----
    @app.route("/api/chats")
    def chats_list():
        d = STORE.get(_flask_g.sid)
        items = sorted(d["chats"].values(), key=lambda c: -c["updated"])
        return jsonify({"success": True, "active": d["active"],
                        "chats": [{"id": c["id"], "title": c["title"], "updated": c["updated"]} for c in items]})

    @app.route("/api/chats/<cid>")
    def chat_open(cid):
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            c = d["chats"].get(cid)
            if not c:
                return jsonify({"success": False, "error": "Chat not found."}), 404
            d["active"] = cid
            STORE.save(sid, d)
            msgs = list(c["messages"])
        _apply_chat(self.chatbot, c)
        return jsonify({"success": True, "messages": msgs, "title": c["title"]})

    @app.route("/api/chats/new", methods=["POST"])
    def chat_new():
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            d["active"] = None
            STORE.save(sid, d)
        _apply_chat(self.chatbot, None)
        return jsonify({"success": True})

    @app.route("/api/chats/<cid>/rename", methods=["POST"])
    def chat_rename(cid):
        title = " ".join(str((request.get_json(silent=True) or {}).get("title", "")).split())[:60]
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            if cid in d["chats"] and title:
                d["chats"][cid]["title"] = title
                STORE.save(sid, d)
        return jsonify({"success": True})

    @app.route("/api/chats/<cid>", methods=["DELETE"])
    def chat_delete(cid):
        sid = _flask_g.sid
        was_active = False
        with STORE.lock:
            d = STORE.get(sid)
            d["chats"].pop(cid, None)
            if d["active"] == cid:
                d["active"], was_active = None, True
            STORE.save(sid, d)
        if was_active:
            _apply_chat(self.chatbot, None)
        return jsonify({"success": True})

    # ----- math history -----
    @app.route("/api/math_history", methods=["GET"])
    def mh_list():
        return jsonify({"success": True, "items": STORE.get(_flask_g.sid)["math"]})

    @app.route("/api/math_history", methods=["POST"])
    def mh_add():
        if request.content_length and request.content_length > 600000:
            return jsonify({"success": False, "error": "Too big."}), 413
        j = request.get_json(silent=True) or {}
        if j.get("pane") not in ("type", "draw", "image") or not isinstance(j.get("solution"), dict):
            return jsonify({"success": False, "error": "Bad data."}), 400
        item = {"id": uuid.uuid4().hex[:12], "pane": j["pane"], "input": str(j.get("input", ""))[:500],
                "solution": j["solution"], "read": j.get("read") if isinstance(j.get("read"), dict) else None,
                "time": time.time()}
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            d["math"].append(item)
            del d["math"][:-STORE.MAX_MATH]
            STORE.save(sid, d)
        return jsonify({"success": True, "id": item["id"]})

    @app.route("/api/math_history/<mid>", methods=["DELETE"])
    def mh_del(mid):
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            d["math"] = [x for x in d["math"] if x["id"] != mid]
            STORE.save(sid, d)
        return jsonify({"success": True})

    @app.route("/api/math_history/clear", methods=["POST"])
    def mh_clear():
        sid = _flask_g.sid
        with STORE.lock:
            d = STORE.get(sid)
            d["math"] = []
            STORE.save(sid, d)
        return jsonify({"success": True})


_mu_prev_init = WebUI.__init__


def _mu_init(self):
    global SESSIONS
    _mu_prev_init(self)
    SESSIONS = _Sessions(self.chatbot)       # the old shared bot becomes the "template"
    proxy = _BotProxy()
    self.chatbot = proxy
    self.image_analyzer.chatbot = proxy
    self.fake_news_detector.chatbot = proxy
    self.math_reader.bot = proxy
    if FLASK_AVAILABLE:
        _mu_add_routes(self)


WebUI.__init__ = _mu_init

# ---- 5) MATH STUDIO: each pane keeps its own answer + saved history ----
MX_JS = _swap(MX_JS, "const out = $('calc-output');", r"""const out = $('calc-output');
  let curPane = 'type', opPane = 'type', lastRead = null;
  const stash = {};
  function switchOut(name) {                       // move the visible answer away, bring the other pane's answer back
    if (name === curPane) return;
    const frag = document.createDocumentFragment();
    while (out.firstChild) frag.appendChild(out.firstChild);
    stash[curPane] = frag;
    curPane = name;
    if (stash[name]) { out.appendChild(stash[name]); stash[name] = null; }
  }
  function saveHist(d, text) {
    const pane = opPane;
    let read = null;
    if (pane !== 'type' && lastRead && lastRead.chosen) {
      read = Object.assign({}, lastRead, { chosen: Object.assign({}, lastRead.chosen, { latex: String(text || lastRead.chosen.latex) }) });
    }
    fetch('/api/math_history', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pane: pane, input: String(text || '').slice(0, 500), solution: d, read: read }) })
      .then(() => refreshHist()).catch(() => {});
  }""", "math out")
MX_JS = _swap(MX_JS, "function setMode(name) {", "function setMode(name) {\n    switchOut(name);", "setMode")
MX_JS = _swap(MX_JS, "async function solveText(text) {", "async function solveText(text) {\n    opPane = curPane;", "solveText")
MX_JS = _swap(MX_JS, "if (busy) return; busy = true;", "if (busy) return; busy = true; opPane = curPane;", "readBlob busy")
MX_JS = _swap(MX_JS, "function showSolution(d, problemText) {",
              "function showSolution(d, problemText, noSave) {\n    if (opPane !== curPane) setMode(opPane);\n    if (!noSave) saveHist(d, problemText);", "showSolution")
MX_JS = _swap(MX_JS, "function showError(msg) {", "function showError(msg) {\n    if (opPane !== curPane) setMode(opPane);", "showError")
MX_JS = _swap(MX_JS, "const res = await pollJob('/api/math_read_start', { method: 'POST', body: fd });",
              "const res = await pollJob('/api/math_read_start', { method: 'POST', body: fd });\n      if (opPane !== curPane) setMode(opPane);", "read finished")
MX_JS = _swap(MX_JS, "$('mx-readbox').innerHTML = renderRead(res.read);",
              "lastRead = res.read; $('mx-readbox').innerHTML = renderRead(res.read);", "lastRead")
MX_JS = _swap(MX_JS, r'''stop(); out.innerHTML = '<div class="cr-status warn">''',
              r'''stop(); if (opPane !== curPane) setMode(opPane); out.innerHTML = '<div class="cr-status warn">''', "read error")
MX_JS = _swap(MX_JS, "// ---------------- image upload (file, camera, drag and drop, paste) ----------------", r"""// ---------------- saved math history ----------------
  const mxModes = document.querySelector('.mx-modes');
  const histBtn = document.createElement('button');
  histBtn.type = 'button'; histBtn.className = 'mx-mode'; histBtn.textContent = '🕘 History';
  mxModes.appendChild(histBtn);
  const histBox = document.createElement('div');
  histBox.id = 'mx-hist'; histBox.hidden = true;
  mxModes.insertAdjacentElement('afterend', histBox);
  const PICON = { type: '⌨️', draw: '✍️', image: '📷' };
  let histItems = [];
  function renderHist() {
    if (!histItems.length) { histBox.innerHTML = '<div class="ppl-note">Nothing saved yet. Solved problems appear here.</div>'; return; }
    histBox.innerHTML = '<div class="oc-row" style="margin:0 0 8px"><button type="button" class="oc-btn" data-act="clear">🗑 Clear all</button></div>' +
      histItems.slice().reverse().map(it => '<div class="mh-item" data-id="' + esc(it.id) + '"><span>' + (PICON[it.pane] || '') +
        '</span><span class="mh-in">' + esc(it.input || '…') + '</span><small>' +
        esc(new Date(it.time * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })) +
        '</small><button type="button" class="oc-btn" data-act="del">✖</button></div>').join('');
  }
  async function refreshHist() {
    try { const d = await (await fetch('/api/math_history')).json(); histItems = d.items || []; renderHist(); } catch (e) {}
    return histItems;
  }
  function restore(it) {
    setMode(it.pane); opPane = it.pane;
    skeleton();
    if (it.pane === 'type') inp.value = it.input || '';
    if (it.read && it.read.chosen) { $('mx-readbox').innerHTML = renderRead(it.read); wireRead(it.read); }
    showSolution(it.solution, it.input, true);
  }
  histBox.addEventListener('click', async e => {
    const act = e.target.closest('[data-act]'), row = e.target.closest('.mh-item');
    if (act && act.dataset.act === 'clear') { await fetch('/api/math_history/clear', { method: 'POST' }); await refreshHist(); return; }
    if (!row) return;
    if (act && act.dataset.act === 'del') { await fetch('/api/math_history/' + row.dataset.id, { method: 'DELETE' }); await refreshHist(); return; }
    const it = histItems.find(x => x.id === row.dataset.id);
    if (it) { restore(it); histBox.hidden = true; }
  });
  histBtn.addEventListener('click', () => { histBox.hidden = !histBox.hidden; if (!histBox.hidden) refreshHist(); });
  refreshHist().then(items => {                    // after a refresh, bring back the latest answer of every pane
    ['image', 'draw', 'type'].forEach(p => { const last = items.filter(x => x.pane === p).pop(); if (last) restore(last); });
    setMode('type');
  });

  // ---------------- image upload (file, camera, drag and drop, paste) ----------------""", "history block")

# ---- 6) CHAT HISTORY UI ----
CHIST_CSS = r"""
#chatHist{position:absolute;inset:0;z-index:6;padding:14px;overflow-y:auto;border-radius:20px;background:rgba(6,8,16,.96)}
#chatHist[hidden],#mx-hist[hidden]{display:none}
.ch-item,.mh-item{display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:12px;cursor:pointer;margin-bottom:6px;border:1px solid rgba(221,132,72,.35)}
.ch-item:hover,.ch-item.cur,.mh-item:hover{background:rgba(221,132,72,.25)}
.ch-t,.mh-in{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mh-in{font-family:'Courier New',monospace}
.ch-item small,.mh-item small{opacity:.7;white-space:nowrap}
.ch-item button{background:none;border:0;cursor:pointer;font-size:15px;color:inherit}
#mx-hist{margin:0 0 16px;padding:10px;border-radius:14px;background:rgba(0,0,0,.35);border:1px solid rgba(221,132,72,.5);max-height:260px;overflow-y:auto}
body.light #chatHist{background:rgba(255,255,255,.96);color:#12304f}
body.light #mx-hist{background:rgba(255,255,255,.6)}
"""

CHIST_JS = r"""
(function () {
  const head = document.querySelector('#chat .chat-head');
  const frame = document.querySelector('#chat .chat-frame');
  const wrap = head && head.querySelector('div');
  if (!head || !frame || !wrap) return;
  const esc = s => escapeHtml(String(s == null ? '' : s));

  const btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'mic-btn mini'; btn.textContent = '🕘'; btn.title = 'Chat history';
  wrap.insertBefore(btn, wrap.firstChild);
  const lbl = head.querySelector('.generate-button span'); if (lbl) lbl.textContent = 'New chat';

  const panel = document.createElement('div');
  panel.id = 'chatHist'; panel.hidden = true; frame.appendChild(panel);

  function showMessages(msgs) {
    chatMessages.querySelectorAll('.msg-row').forEach(r => r.remove());
    if (!msgs.length) { chatWelcome.style.display = ''; chatMessages.scrollTop = 0; return; }
    msgs.forEach(m => {
      const row = addRow(m.role === 'user' ? 'user' : 'assistant');
      if (m.role === 'user') row.bubble.textContent = m.text; else row.bubble.innerHTML = renderMarkdown(m.text);
    });
    scrollChat();
  }
  async function openChat(id) {
    if (chatBusy) return;
    try {
      const d = await (await fetch('/api/chats/' + id)).json();
      if (d.success) { showMessages(d.messages); panel.hidden = true; }
    } catch (e) {}
  }
  async function newChat() {
    if (chatBusy) return;
    try { await fetch('/api/chats/new', { method: 'POST' }); } catch (e) {}
    showMessages([]); panel.hidden = true;
  }
  window.clearChat = newChat;                       // the old "Clear" button now starts a new chat

  async function renderPanel() {
    let d; try { d = await (await fetch('/api/chats')).json(); } catch (e) { return; }
    panel.innerHTML = '<div class="oc-row" style="margin:0 0 10px"><button type="button" class="oc-btn" data-act="new">➕ New chat</button></div>' +
      (d.chats.length ? d.chats.map(c => '<div class="ch-item' + (c.id === d.active ? ' cur' : '') + '" data-id="' + esc(c.id) + '">' +
        '<span class="ch-t">' + esc(c.title) + '</span><small>' +
        esc(new Date(c.updated * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })) +
        '</small><button type="button" data-act="ren" title="Rename">✏️</button><button type="button" data-act="del" title="Delete">🗑</button></div>').join('')
        : '<div class="ppl-note">No saved chats yet. Send a message and it is saved automatically.</div>');
  }
  panel.addEventListener('click', async e => {
    const act = e.target.closest('[data-act]'), row = e.target.closest('.ch-item');
    if (act && act.dataset.act === 'new') return newChat();
    if (!row) return;
    const id = row.dataset.id;
    if (act && act.dataset.act === 'ren') {
      const t = prompt('Rename chat', row.querySelector('.ch-t').textContent);
      if (t && t.trim()) {
        await fetch('/api/chats/' + id + '/rename', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title: t }) });
        renderPanel();
      }
      return;
    }
    if (act && act.dataset.act === 'del') {
      if (!confirm('Delete this chat?')) return;
      const wasCur = row.classList.contains('cur');
      await fetch('/api/chats/' + id, { method: 'DELETE' });
      if (wasCur) showMessages([]);
      renderPanel();
      return;
    }
    openChat(id);
  });
  btn.addEventListener('click', () => { panel.hidden = !panel.hidden; if (!panel.hidden) renderPanel(); });

  fetch('/api/chats').then(r => r.json()).then(d => { if (d.active) openChat(d.active); }).catch(() => {});
})();
"""

_mu_prev_get_html = WebUI.get_html


def _mu_get_html(self) -> str:
    html = _mu_prev_get_html(self)
    html = html.replace("</head>", "<style>" + CHIST_CSS + "</style></head>", 1)
    return html.replace("</body>", "<script>" + CHIST_JS + "</script></body>", 1)


WebUI.get_html = _mu_get_html

# ============================================================================
# CODE PROJECT PATCH: upload files / folders / ZIP / TAR, any file size
# Paste ABOVE the MAIN EXECUTION block, after `WebUI.get_html = _mu_get_html`
# ============================================================================
import zipfile
import tarfile
import posixpath
import bisect

_CP_TL = threading.local()

_CP_EXT = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".java": "java",
    ".c": "c", ".h": "cpp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby",
    ".kt": "kotlin", ".kts": "kotlin", ".swift": "swift", ".scala": "scala", ".lua": "lua",
    ".dart": "dart", ".sh": "bash", ".bash": "bash", ".html": "html", ".htm": "html", ".css": "css",
}
_CP_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv", "env", "dist", "build",
                 ".idea", ".vscode", "site-packages", "target", "vendor", ".next", "__macosx",
                 ".mypy_cache", ".pytest_cache", ".gradle", ".tox"}
_CP_MAX_FILE = 8 * 1024 * 1024            # biggest single code file
_CP_MAX_ARCHIVE = 100 * 1024 * 1024       # biggest archive inside an archive
_CP_MAX_TOTAL = 300 * 1024 * 1024         # everything unpacked
_CP_MAX_FILES = int(os.getenv("CODE_MAX_FILES", "400"))
_CP_CHUNK = int(os.getenv("CODE_CHUNK_CHARS", "6000"))
_CP_AI_BUDGET = int(os.getenv("CODE_MAX_AI_CHUNKS", "60"))   # AI-reviewed parts per check

CodeChecker.LANGUAGE_ALIASES.update({v: v for v in set(_CP_EXT.values())})
CodeChecker.LANGUAGE_ALIASES.update({"ts": "typescript", "c#": "csharp", "golang": "go", "rs": "rust",
                                     "rb": "ruby", "kt": "kotlin", "sh": "bash", "shell": "bash"})

# ---- a language the parser does not know must not be reported as a syntax error ----
_cp_orig_syntax = CodeChecker._check_syntax


def _cp_syntax_wrapper(self, code, language):
    errs = _cp_orig_syntax(self, code, language)
    return [e for e in errs if not e.startswith(("Could not analyze this language", "Tree-sitter is not installed"))]


CodeChecker._check_syntax = _cp_syntax_wrapper

# ---- let the model know which part of a big file it is looking at ----
_cp_orig_generate = CodeChecker._generate


def _cp_generate_ctx(self, messages, max_new_tokens):
    ctx = getattr(_CP_TL, "ctx", None)
    if ctx:
        messages = [dict(m) for m in messages]
        for m in messages:
            if m["role"] == "user" and "Code to review:" in m["content"]:
                m["content"] = m["content"].replace("Code to review:", ctx + "\n\nCode to review (this part only):", 1)
                break
    return _cp_orig_generate(self, messages, max_new_tokens)


CodeChecker._generate = _cp_generate_ctx

# ---- a part cut out of a big file may not parse alone: skip the per-part syntax test ----
_cp_orig_rse = CodeChecker._real_syntax_errors


def _cp_rse(self, code, language):
    if getattr(_CP_TL, "partial", False):
        return []
    return _cp_orig_rse(self, code, language)


CodeChecker._real_syntax_errors = _cp_rse

_CP_PARSE = {}


def _cp_can_parse(lang):
    if lang == "python":
        return True
    if lang not in _CP_PARSE:
        try:
            from tree_sitter_language_pack import get_parser
            get_parser(lang)
            _CP_PARSE[lang] = True
        except Exception:
            _CP_PARSE[lang] = False
    return _CP_PARSE[lang]


# ---------------------------------------------------------------- reading uploads
def _cp_clean_path(p):
    p = (p or "").replace("\\", "/").strip("/")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or ".." in parts or re.match(r"^[A-Za-z]:$", parts[0]):
        return ""
    return "/".join(parts)[:300]


def _cp_decode(b):
    if b.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return b.decode("utf-16")
        except Exception:
            return None
    if b"\x00" in b[:8192]:
        return None
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:
        return b.decode("cp1252", errors="replace")


def _cp_lang(path, text):
    ext = posixpath.splitext(path.lower())[1]
    if ext in _CP_EXT:
        return _CP_EXT[ext]
    if not ext:
        first = text.split("\n", 1)[0]
        if first.startswith("#!"):
            for key, val in (("python", "python"), ("node", "javascript"), ("bash", "bash"),
                             ("/sh", "bash"), ("ruby", "ruby"), ("php", "php")):
                if key in first:
                    return val
    return None


_CP_ARCH = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _cp_collect(uploads, cancel=None):
    """uploads: [(path, bytes)] -> (files [(path, text, lang)], skipped [{path, reason}], ignored_count)"""
    cands, skipped = {}, []
    st = {"total": 0, "ignored": 0, "stop": False}

    def cancelled():
        try:
            return bool(cancel and cancel())
        except Exception:
            return False

    def skip(p, why):
        if len(skipped) < 500:
            skipped.append({"path": p, "reason": why})

    def want(inner, size):
        parts = inner.split("/")
        low = inner.lower()
        if any(x.lower() in _CP_SKIP_DIRS for x in parts[:-1]):
            return False
        if low.endswith(_CP_ARCH):
            return True
        ext = posixpath.splitext(low)[1]
        return ext in _CP_EXT or (ext == "" and size < 200000)

    def handle(path, data, depth):
        path = _cp_clean_path(path)
        if not path or st["stop"] or cancelled():
            st["stop"] = st["stop"] or cancelled()
            return
        low = path.lower()
        parts = path.split("/")
        if any(x.lower() in _CP_SKIP_DIRS for x in parts[:-1]):
            st["ignored"] += 1
            return
        if low.endswith(".zip"):
            if depth >= 3:
                return skip(path, "archives nested too deeply")
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        if info.is_dir() or st["stop"] or cancelled():
                            continue
                        inner = _cp_clean_path(info.filename)
                        if not inner:
                            continue
                        ip = path + "/" + inner
                        if not want(inner, info.file_size):
                            st["ignored"] += 1
                            continue
                        lim = _CP_MAX_ARCHIVE if inner.lower().endswith(_CP_ARCH) else _CP_MAX_FILE
                        if info.file_size > lim:
                            skip(ip, f"larger than {lim // 1048576} MB")
                            continue
                        st["total"] += info.file_size
                        if st["total"] > _CP_MAX_TOTAL:
                            skip(ip, "total size limit reached, the rest was not read")
                            st["stop"] = True
                            break
                        try:
                            handle(ip, z.read(info), depth + 1)
                        except Exception:
                            skip(ip, "encrypted or unreadable")
            except Exception as e:
                skip(path, "could not open the ZIP: " + str(e)[:80])
            return
        if low.endswith(_CP_ARCH):
            if depth >= 3:
                return skip(path, "archives nested too deeply")
            try:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                    for m in tf:
                        if not m.isfile() or st["stop"] or cancelled():
                            continue
                        inner = _cp_clean_path(m.name)
                        if not inner:
                            continue
                        ip = path + "/" + inner
                        if not want(inner, m.size):
                            st["ignored"] += 1
                            continue
                        lim = _CP_MAX_ARCHIVE if inner.lower().endswith(_CP_ARCH) else _CP_MAX_FILE
                        if m.size > lim:
                            skip(ip, f"larger than {lim // 1048576} MB")
                            continue
                        st["total"] += m.size
                        if st["total"] > _CP_MAX_TOTAL:
                            skip(ip, "total size limit reached, the rest was not read")
                            st["stop"] = True
                            break
                        fobj = tf.extractfile(m)
                        if fobj:
                            handle(ip, fobj.read(), depth + 1)
            except Exception as e:
                skip(path, "could not open the archive: " + str(e)[:80])
            return
        ext = posixpath.splitext(low)[1]
        if ext not in _CP_EXT and ext != "":
            st["ignored"] += 1
            return
        if len(data) > _CP_MAX_FILE:
            return skip(path, "larger than 8 MB")
        cands[path] = data

    for name, data in uploads:
        if cancelled():
            st["stop"] = True
            break
        handle(name, data, 0)

    files = []
    for path in sorted(cands):
        text = _cp_decode(cands[path])
        if text is None:
            skip(path, "binary file")
            continue
        lang = _cp_lang(path, text)
        if not lang:
            st["ignored"] += 1
            continue
        lines = text.count("\n") + 1
        if ".min." in path.lower() or (len(text) > 20000 and len(text) / lines > 400):
            skip(path, "minified or generated code")
            continue
        if len(files) >= _CP_MAX_FILES:
            skip(path, f"more than {_CP_MAX_FILES} files (set CODE_MAX_FILES to raise the limit)")
            continue
        files.append((path, text, lang))
    return files, skipped, st["ignored"]


# ---------------------------------------------------------------- splitting big files
def _cp_split(code, lang, limit):
    """Returns [(start_line, end_line)] parts, each at most `limit` characters, cut at function/class borders."""
    lines = code.splitlines(keepends=True)
    n = len(lines)
    if len(code) <= limit or n <= 1:
        return [(0, n)]
    cand = set()
    if lang == "python":
        try:
            for node in ast.parse(code).body:
                s = node.lineno - 1
                for d in (getattr(node, "decorator_list", None) or []):
                    s = min(s, d.lineno - 1)
                if s > 0:
                    cand.add(s)
        except SyntaxError:
            pass
    if not cand:
        closing = ("}", ")", "]", "else", "elif", "except", "finally", "catch", "case ", "default")
        c1, c0 = set(), set()
        for i in range(1, n):
            ln = lines[i]
            if ln.strip() and not ln[0].isspace() and not ln.lstrip().startswith(closing):
                c0.add(i)
                if not lines[i - 1].strip():
                    c1.add(i)
        cand = c1 or c0
    cl = sorted(cand)
    cum = [0]
    for l in lines:
        cum.append(cum[-1] + len(l))
    out, start = [], 0
    while start < n:
        if cum[n] - cum[start] <= limit:
            out.append((start, n))
            break
        lim_idx = bisect.bisect_right(cum, cum[start] + limit) - 1
        j = bisect.bisect_right(cl, lim_idx) - 1
        cut = cl[j] if (j >= 0 and cl[j] > start) else max(start + 1, lim_idx)
        out.append((start, cut))
        start = cut
    return out


def _cp_outline(code, lang):
    if lang != "python":
        return "", ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", ""
    names = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(f"def {n.name}({', '.join(a.arg for a in n.args.args)})")
        elif isinstance(n, ast.ClassDef):
            meths = [m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            names.append(f"class {n.name}" + (f" (methods: {', '.join(meths[:15])})" if meths else ""))
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
    imports = "\n".join(l.strip() for l in code.splitlines() if re.match(r"\s*(import|from)\s+\S", l))
    return "; ".join(names)[:1500], imports[:1500]


# ---------------------------------------------------------------- checking one file
def _cp_big(checker, job, stats, budget, step, path, code, lang, syn, issues, chunks, review=None):
    lines = code.splitlines(keepends=True)
    outline, imports = _cp_outline(code, lang)
    parts = ["".join(lines[a:b]) for a, b in chunks]
    new_parts = list(parts)
    partial_idx, problems, why, other, notes = set(), [], [], [], []
    reviewed = skipped = failed = 0
    N = len(chunks)
    for k, (a, b) in enumerate(chunks):
        text = parts[k]
        if job["cancel"]:
            skipped += N - k
            notes.append("Stopped by you before every part was reviewed.")
            break
        if budget["left"] <= 0:
            skipped += 1
            step(path, f"{path}: part {k + 1}/{N} skipped")
            continue
        budget["left"] -= 1
        alone_ok = lang == "python" and not checker._check_syntax(text, "python")
        ctx = (f"This is part {k + 1} of {N} of the file '{path}'. Other parts are reviewed separately, so do NOT "
               "report names, imports or helper functions as missing if they could be defined in another part. "
               "Line numbers in your answer must count from the first line shown below (that line is line 1).")
        if outline:
            ctx += "\nNames defined in the whole file: " + outline
        if imports:
            ctx += "\nImports of the whole file:\n" + imports
        if syn:
            ctx += "\nThe parser reported these errors somewhere in the whole file:\n" + "\n".join(syn[:5])
        c_issues = checker._find_python_issues(text) if alone_ok else []
        _CP_TL.ctx, _CP_TL.partial = ctx, not alone_ok
        try:
            if review is not None:
                res = review(text, lang, [], c_issues)
            else:
                res = checker._fix_with_model(text, lang, [], c_issues)
        except Exception as e:
            print("Code part error:", str(e)[:150])
            res = None
        finally:
            _CP_TL.ctx, _CP_TL.partial = None, False
        step(path, f"{path}: part {k + 1}/{N}")
        if res is None:
            failed += 1
            continue
        reviewed += 1
        if res["status"] == "correct":
            continue
        for p in res["problems"]:
            if re.match(r"\s*no problems found", p, re.I):
                continue
            problems.append(re.sub(r"\b([Ll]ines?) (\d+)", lambda m: f"{m.group(1)} {int(m.group(2)) + a}", p, count=1))
        if res["status"] == "fixed" and res["fixed_code"].strip() != text.strip():
            tail = text[len(text.rstrip()):]
            new_parts[k] = res["fixed_code"].rstrip() + tail
            why.extend(res["why_correct"])
            for o in res.get("other_ways") or []:
                if str(o) not in other:
                    other.append(str(o))
            if not alone_ok:
                partial_idx.add(k)

    changed = new_parts != parts
    verified = None
    if changed:
        if _cp_can_parse(lang):
            before = len(checker._real_syntax_errors(code, lang))

            def ok(ps):
                return len(checker._real_syntax_errors("".join(ps), lang)) <= before
            if ok(new_parts):
                verified = True
            else:
                alt = [parts[i] if i in partial_idx else new_parts[i] for i in range(N)]
                if alt != new_parts and alt != parts and ok(alt):
                    new_parts, verified = alt, True
                    notes.append("Some part fixes were dropped because they broke the file.")
                else:
                    new_parts = list(parts)
                    notes.append("The combined fixes did not parse, so they were NOT applied. Use the problem list to fix it by hand.")
            changed = new_parts != parts
        else:
            verified = False
    stats["parts_total"] += N
    stats["parts_reviewed"] += reviewed
    stats["parts_skipped"] += skipped + failed
    if skipped or failed:
        notes.append(f"{reviewed} of {N} parts were reviewed by the AI ({skipped} skipped by the limit, {failed} without a usable answer).")
    if not problems and not syn and not issues:
        status = "correct" if (skipped == 0 and failed == 0) else "scanned"
    elif changed:
        status = "fixed" if verified else "fixed_unverified"
    else:
        status = "suggestion_only"
    return {"syntax_errors": syn, "issues": issues, "problems": problems, "fixed_code": "".join(new_parts),
            "why_correct": why, "other_ways": other[:20], "status": status, "verified": verified,
            "changed": changed, "notes": notes}


def _cp_rules(checker, code, lang, syn, issues, note=None):
    fx = checker._fix_with_rules(code, lang, syn, issues)
    changed = fx["fixed_code"].strip() != code.strip()
    st = fx["status"]
    why = fx["why_correct"]
    if not changed:
        st = "scanned" if not fx["problems"] else "suggestion_only"
        why = []
    return {"syntax_errors": syn, "issues": issues, "problems": fx["problems"], "fixed_code": fx["fixed_code"],
            "why_correct": why, "other_ways": list(fx.get("other_ways") or []), "status": st,
            "verified": fx["verified"], "changed": changed,
            "notes": [note] if note else []}


def _cp_entry(path, lang, code, r, fixed):
    problems = [p for p in (r.get("problems") or []) if not re.match(r"\s*no problems found", p, re.I)]
    changed = bool(r.get("changed")) and (r.get("fixed_code") or "").strip() != code.strip()
    if changed:
        fixed[path] = r["fixed_code"]
    return {"path": path, "language": lang, "lines": code.count("\n") + 1, "chars": len(code),
            "status": r.get("status", "scanned"), "syntax_errors": list(r.get("syntax_errors") or [])[:10],
            "issues": [i for i in (r.get("issues") or []) if "TODO or FIXME" not in i][:30],
            "problems": problems[:60], "problems_total": len(problems),
            "tips": list(r.get("tips") or [])[:40],
            "analysis": list(r.get("analysis") or [])[:20],
            "other_ways": list(r.get("other_ways") or [])[:20],
            "why_better": list(r.get("why_better") or [])[:20],
            "why_correct": list(r.get("why_correct") or [])[:40], "changed": changed,
            "kind": r.get("kind") or "fix",
            "verified": r.get("verified"), "notes": list(r.get("notes") or [])}


def _cp_report_md(res):
    s = res.get("summary", {})
    L = ["# pip ai agent: code check report", "",
         f"Files checked: {s.get('checked')}  |  with problems: {s.get('with_problems')}  |  fixed: {s.get('fixed')}  |  skipped: {s.get('skipped')}", ""]
    for f in res.get("files", []):
        if not (f["problems"] or f["syntax_errors"] or f["issues"]):
            continue
        L += ["", f"## {f['path']}  ({f['language']}, {f['lines']} lines, status: {f['status']})"]
        if f["syntax_errors"]:
            L += ["", "Syntax errors:"] + ["- " + e for e in f["syntax_errors"]]
        if f["problems"]:
            L += ["", "Problems:"] + ["- " + e for e in f["problems"]]
        elif f["issues"]:
            L += ["", "Static checks:"] + ["- " + e for e in f["issues"]]
        if f["why_correct"]:
            L += ["", "Why the fix is correct:"] + ["- " + e for e in f["why_correct"]]
    if res.get("skipped"):
        L += ["", "## Skipped files"] + [f"- {x['path']}: {x['reason']}" for x in res["skipped"]]
    return "\n".join(L)


# ---------------------------------------------------------------- Gemini fast review
# The local HF model is slow on CPU (minutes per file). When the Gemini key is
# available we review through the API instead: same prompt, same parser, same
# verification - just seconds per file. Falls back to the HF model on failure.
def _cp_gemini_ok(ui):
    bot = getattr(ui, "chatbot", None)
    if bot is None:
        return None
    try:
        return bot if bot._ensure_client() else None
    except Exception:
        return None


def _cp_gemini_generate(bot, prompt, max_tokens):
    cands = []
    for m in [getattr(bot, "chat_model", None)] + list(getattr(bot, "CHAT_MODELS", []) or []):
        if m and m not in cands:
            cands.append(m)
    if not cands:
        return ""
    cfg = genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=max_tokens)
    last = None
    for m in cands[:3]:
        try:
            resp = bot.client.models.generate_content(model=m, contents=prompt, config=cfg)
            txt = (getattr(resp, "text", None) or "").strip()
            if txt:
                bot.chat_model = m
                return txt
            last = f"{m}: empty answer"
        except Exception as e:
            last = f"{m}: {e}"
    if last:
        print("Gemini code review model problem:", str(last)[:150])
    return ""


def _cp_gemini_call(bot, checker, prompt, max_tokens):
    """_cp_gemini_generate that can be cancelled mid-request: the HTTP call runs
    on a daemon thread while we poll the job's cancel flag every 0.3s. On cancel
    we return "" right away; the orphaned thread finishes harmlessly in the
    background (its result is simply discarded)."""
    chk = getattr(checker, "_cancel_check", None)
    if chk is None:
        return _cp_gemini_generate(bot, prompt, max_tokens)
    import queue as _q
    q = _q.Queue(maxsize=1)

    def work():
        try:
            q.put(_cp_gemini_generate(bot, prompt, max_tokens))
        except Exception:
            q.put("")

    threading.Thread(target=work, daemon=True).start()
    while True:
        try:
            return q.get(timeout=0.3)
        except _q.Empty:
            try:
                if chk():
                    return ""
            except Exception:
                return ""


def _cp_gemini_review(bot, checker, code, lang, syn, issues):
    """Same contract as CodeChecker._fix_with_model, via Gemini. None on failure."""
    try:
        parts = [f"Language: {lang}", "", "Code to review:", f"```{lang}", code, "```", ""]
        if syn:
            parts.append("The parser already reported these syntax errors (they are real):")
            parts += [f"- {e}" for e in syn[:8]]
            parts.append("")
        if issues:
            parts.append("Static analysis also flagged:")
            parts += [f"- {i}" for i in issues[:8]]
            parts.append("")
        parts.append(checker.OUTPUT_FORMAT)
        reply = _cp_gemini_call(bot, checker, "\n".join(parts), min(8000, max(2500, len(code) // 2)))
        if not reply:
            return None
        parsed = checker._parse_reply(reply, code)
        if parsed is None:
            return None
        if parsed["verdict"] == "correct" and not syn:
            parsed["fixed_code"] = code
            parsed["verified"] = True
        else:
            parsed["verified"] = not checker._real_syntax_errors(parsed["fixed_code"], lang)
        changed = parsed["fixed_code"].strip() != code.strip()
        if parsed["verdict"] == "correct" and not syn:
            status = "correct"
        elif not changed:
            status = "suggestion_only"
        elif parsed["verified"]:
            status = "fixed"
        else:
            status = "fixed_unverified"
        return {"problems": parsed["problems"], "fixed_code": parsed["fixed_code"],
                "why_correct": parsed["why_correct"], "other_ways": parsed.get("other_ways") or [],
                "status": status, "verified": parsed["verified"]}
    except Exception as e:
        print("Gemini code review failed:", str(e)[:150])
        return None


def _cp_gemini_tips(bot, checker, code, lang, issues):
    """Full improvement review via Gemini. Returns the rich dict, or None."""
    try:
        head = []
        c = getattr(_CP_TL, "ctx", None)
        if c:
            head.append(c + "\n\nLine numbers count from the first line shown (that line is line 1).")
        parts = head + [f"Language: {lang}", "", "Code to review:", f"```{lang}", code, "```", ""]
        if issues:
            parts.append("Static analysis also flagged:")
            parts += [f"- {i}" for i in issues[:8]]
            parts.append("")
        parts.append(checker.IMPROVE_FORMAT)
        reply = _cp_gemini_call(bot, checker, "\n".join(parts), min(9000, max(3000, len(code) // 2)))
        if not reply:
            return None
        verdict = (checker._section(reply, "verdict") or "").lower()
        analysis = checker._bullets(checker._section(reply, "analysis") or "")
        tips = checker._bullets(checker._section(reply, "tips") or "")
        other = [o for o in checker._bullets(checker._section(reply, "other_ways") or "")
                 if not re.match(r"(?i)^none\b", o.strip())]
        why = checker._bullets(checker._section(reply, "why_better") or "")
        imp = checker._strip_fences((checker._section(reply, "improved_code") or "").strip())
        clean = "clean" in verdict or (not tips and not analysis)
        if clean:
            return {"tips": [], "analysis": [], "other_ways": [], "why_better": [],
                    "improved_code": code, "status": "clean"}
        return {"tips": tips, "analysis": analysis, "other_ways": other, "why_better": why,
                "improved_code": imp if imp.strip() else code,
                "status": "improved" if imp.strip() and imp.strip() != code.strip() else "tips"}
    except Exception as e:
        print("Gemini tips review failed:", str(e)[:150])
        return None


def _cp_big_tips(checker, bot, gemini, job, stats, budget, step, path, code, lang, syn, issues, chunks):
    """Improve-mode version of _cp_big: analysis + tips + other ways per part,
    and the improved code of each part is joined back into a full file."""
    lines = code.splitlines(keepends=True)
    outline, imports = _cp_outline(code, lang)
    parts = ["".join(lines[a:b]) for a, b in chunks]
    new_parts = list(parts)
    tips, analysis, other, why_b, notes = [], [], [], [], []
    N = len(chunks)
    reviewed = skipped = failed = 0
    for k, (a, b) in enumerate(chunks):
        text = parts[k]
        if job["cancel"]:
            skipped += N - k
            notes.append("Stopped by you before every part was reviewed.")
            break
        if budget["left"] <= 0:
            skipped += 1
            step(path, f"{path}: part {k + 1}/{N} skipped")
            continue
        budget["left"] -= 1
        alone_ok = lang == "python" and not checker._check_syntax(text, "python")
        ctx = (f"This is part {k + 1} of {N} of the file '{path}'. Other parts are reviewed separately, so do NOT "
               "report names, imports or helper functions as missing if they could be defined in another part. "
               "Line numbers in your answer must count from the first line shown below (that line is line 1). "
               "improved_code must contain ONLY this part's improved code, never the whole file.")
        if outline:
            ctx += "\nNames defined in the whole file: " + outline
        if imports:
            ctx += "\nImports of the whole file:\n" + imports
        c_issues = checker._find_python_issues(text) if alone_ok else []
        _CP_TL.ctx, _CP_TL.partial = ctx, not alone_ok
        try:
            if gemini is not None:
                res = _cp_gemini_tips(gemini, checker, text, lang, c_issues)
            else:
                res = checker._improve_with_model(text, lang, c_issues)
        except Exception as e:
            print("Improve part error:", str(e)[:150])
            res = None
        finally:
            _CP_TL.ctx, _CP_TL.partial = None, False
        step(path, f"{path}: part {k + 1}/{N}")
        if res is None:
            failed += 1
            continue
        reviewed += 1
        for t in res.get("tips") or []:
            if re.match(r"\s*no improvement", t, re.I):
                continue
            tips.append(re.sub(r"\b([Ll]ines?) (\d+)", lambda m: f"{m.group(1)} {int(m.group(2)) + a}", t, count=1))
        for coll, src in ((analysis, res.get("analysis")), (other, res.get("other_ways")), (why_b, res.get("why_better"))):
            for x in src or []:
                if not re.match(r"(?i)^\s*-?\s*(none|no improvement)\b", str(x)) and str(x) not in coll:
                    coll.append(str(x))
        imp = (res.get("improved_code") or "").strip()
        if imp and imp != text.strip() and not imp.startswith("```"):
            tail = text[len(text.rstrip()):]
            new_parts[k] = imp.rstrip() + tail
    stats["parts_total"] += N
    stats["parts_reviewed"] += reviewed
    stats["parts_skipped"] += skipped + failed
    if skipped or failed:
        notes.append(f"{reviewed} of {N} parts were reviewed by the AI ({skipped} skipped by the limit, {failed} without a usable answer).")
    improved = new_parts != parts
    status = "improved" if improved else ("tips" if tips else ("scanned" if reviewed else "suggestion_only"))
    return {"syntax_errors": syn, "issues": issues, "tips": tips[:40], "analysis": analysis[:20],
            "other_ways": other[:20], "why_better": why_b[:20],
            "fixed_code": "".join(new_parts) if improved else code,
            "changed": improved, "status": status, "kind": "improve", "notes": notes}


# ---------------------------------------------------------------- the job
CODE_JOBS = {}
_CP_RUN = threading.Lock()


def _cp_run(ui, job, uploads, ai):
    checker = ui.code_checker
    checker._gemini_bot = None  # _cp_run handles Gemini itself; don't let check_code retry it
    checker._cancel_check = lambda: bool(job.get("cancel"))
    mode = (job.get("mode") or "fix").lower()
    gemini = _cp_gemini_ok(ui) if ai else None

    def prog(stage, pct=None, file=""):
        job["stage"], job["file"] = stage, file
        if pct is not None:
            job["pct"] = max(job["pct"], min(99, int(pct)))

    prog("Waiting for another check to finish")
    with _CP_RUN:
        prog("Opening the files", 3)
        files, skipped, ignored = _cp_collect(uploads, lambda: job.get("cancel"))
        if job["cancel"]:
            job["result"] = {"success": True, "mode": mode, "files": [], "skipped": skipped,
                             "summary": {"checked": 0, "with_problems": 0, "fixed": 0, "skipped": len(skipped),
                                         "ignored": ignored, "lines": 0, "ai": False,
                                         "parts_total": 0, "parts_reviewed": 0, "parts_skipped": 0},
                             "notes": ["Stopped by you while opening the files."], "cancelled": True}
            return
        if not files:
            why = "; ".join(s["reason"] for s in skipped[:3])
            job["result"] = {"success": False, "error": "I found no code files to check." + (" (" + why + ")" if why else "")}
            return
        use_ai = bool(ai and (checker.model is not None or gemini is not None))
        pre = []
        pre_cancelled = False
        for i, (path, text, lang) in enumerate(files):
            if job["cancel"]:
                pre_cancelled = True
                for j in range(i, len(files)):
                    skipped.append({"path": files[j][0], "reason": "not scanned, you stopped the check"})
                break
            prog("Scanning syntax and safety rules", 5 + 10 * i / len(files), path)
            syn = checker._check_syntax(text, lang)
            issues = checker._find_python_issues(text) if (lang == "python" and not syn) else []
            big = len(text) > checker.MAX_CHARS
            chunks = _cp_split(text, lang, _CP_CHUNK) if (use_ai and big) else None
            pre.append({"path": path, "text": text, "lang": lang, "syn": syn, "issues": issues, "chunks": chunks})
        if pre_cancelled:
            job["fixed"] = {}
            job["result"] = {"success": True, "mode": mode, "files": [], "skipped": skipped,
                             "summary": {"checked": 0, "with_problems": 0, "fixed": 0, "skipped": len(skipped),
                                         "ignored": ignored, "lines": 0, "ai": use_ai,
                                         "parts_total": 0, "parts_reviewed": 0, "parts_skipped": 0},
                             "notes": ["Stopped by you during the scan."], "cancelled": True}
            return
        order = sorted(range(len(pre)), key=lambda i: (0 if (pre[i]["syn"] or pre[i]["issues"]) else 1, pre[i]["path"]))
        units = {"done": 0, "total": max(1, sum(len(p["chunks"]) if p["chunks"] else 1 for p in pre))}

        def step(path, label="Reviewing"):
            units["done"] += 1
            prog("Reviewing with the AI" if use_ai else "Checking", 15 + 84 * units["done"] / units["total"], label)

        budget = {"left": _CP_AI_BUDGET}
        stats = {"parts_total": 0, "parts_reviewed": 0, "parts_skipped": 0}
        entries, fixed, cancelled = [], {}, False
        reached = set()
        for i in order:
            p = pre[i]
            if job["cancel"]:
                cancelled = True
                break
            reached.add(i)
            path, text, lang, syn, issues = p["path"], p["text"], p["lang"], p["syn"], p["issues"]
            prog("Reviewing with the AI" if use_ai else "Checking", 15 + 84 * units["done"] / units["total"], path)
            try:
                if not use_ai:
                    r = _cp_rules(checker, text, lang, syn, issues,
                                  "Fast scan: no AI review." if not ai else "No AI reviewer is available, so only syntax and rule checks ran.")
                    step(path, path)
                elif mode == "improve":
                    if p["chunks"]:
                        r = _cp_big_tips(checker, gemini, job, stats, budget, step, path, text, lang, syn, issues, p["chunks"])
                    elif budget["left"] <= 0:
                        r = {"syntax_errors": syn, "issues": issues, "tips": [], "status": "suggestion_only",
                             "kind": "improve",
                             "notes": ["AI tips skipped (limit of parts per check reached)."]}
                        stats["parts_total"] += 1
                        stats["parts_skipped"] += 1
                        step(path, path)
                    else:
                        budget["left"] -= 1
                        if gemini is not None:
                            tres = _cp_gemini_tips(gemini, checker, text, lang, issues)
                        else:
                            tres = checker._improve_with_model(text, lang, issues)
                        imp = ((tres or {}).get("improved_code") or "").strip()
                        changed = bool(tres) and bool(imp) and imp != text.strip()
                        r = {"syntax_errors": syn, "issues": issues,
                             "tips": (tres or {}).get("tips") or [],
                             "analysis": (tres or {}).get("analysis") or [],
                             "other_ways": (tres or {}).get("other_ways") or [],
                             "why_better": (tres or {}).get("why_better") or [],
                             "fixed_code": imp if changed else text,
                             "changed": changed,
                             "status": "improved" if changed else ("tips" if (tres or {}).get("tips") else ("scanned" if tres else "suggestion_only")),
                             "kind": "improve",
                             "notes": [] if tres else ["This file could not be reviewed for improvements."]}
                        stats["parts_total"] += 1
                        stats["parts_reviewed"] += 1
                        step(path, path)
                elif p["chunks"]:
                    def _rev(t, l, s_, i_):
                        if gemini is not None:
                            rr = _cp_gemini_review(gemini, checker, t, l, s_, i_)
                            if rr is not None:
                                return rr
                        return checker._fix_with_model(t, l, s_, i_)
                    r = _cp_big(checker, job, stats, budget, step, path, text, lang, syn, issues, p["chunks"], review=_rev)
                elif budget["left"] <= 0:
                    r = _cp_rules(checker, text, lang, syn, issues, "AI review skipped (limit of parts per check reached).")
                    stats["parts_total"] += 1
                    stats["parts_skipped"] += 1
                    step(path, path)
                else:
                    budget["left"] -= 1
                    fx = _cp_gemini_review(gemini, checker, text, lang, syn, issues) if gemini is not None else None
                    if fx is not None:
                        r = {"success": True, "language": lang, "syntax_errors": syn, "issues": issues,
                             "problems": fx["problems"], "fixed_code": fx["fixed_code"], "why_correct": fx["why_correct"],
                             "other_ways": fx.get("other_ways") or [],
                             "status": fx["status"], "verified": fx["verified"],
                             "changed": fx["fixed_code"].strip() != text.strip(),
                             "recommendations": [], "analysis_engine": "Gemini API"}
                    else:
                        r = checker.check_code(text, lang)
                        if not r.get("success"):
                            r = _cp_rules(checker, text, lang, syn, issues)
                    stats["parts_total"] += 1
                    stats["parts_reviewed"] += 1
                    step(path, path)
            except Exception as e:
                print("Code file error:", path, str(e)[:150])
                r = {"syntax_errors": syn, "issues": issues, "problems": [], "status": "suggestion_only",
                     "kind": "improve" if mode == "improve" else "fix",
                     "notes": ["This file could not be fully checked: " + str(e)[:100]]}
            entries.append(_cp_entry(path, lang, text, r, fixed))
        if cancelled:
            for i, p in enumerate(pre):
                if i not in reached:
                    skipped.append({"path": p["path"], "reason": "not reached, you stopped the check"})
        entries.sort(key=lambda e: (0 if (e["syntax_errors"] or e["problems"] or e["issues"]) else 1, e["path"]))
        notes = []
        if not ai:
            notes.append("Fast scan: only syntax and safety rules were used. Run it again without fast scan for the AI review.")
        elif not use_ai:
            notes.append("No AI reviewer is available (no Gemini key and no local code model), so only syntax and rule checks ran.")
        elif stats["parts_skipped"]:
            notes.append(f"The AI reviewed {stats['parts_reviewed']} of {stats['parts_total']} parts. The limit is {_CP_AI_BUDGET} parts per check "
                         "(files with errors go first). Set CODE_MAX_AI_CHUNKS to raise it, or upload fewer files.")
        notes.append("Syntax is verified by the parser, but only running tests proves that the logic is right.")
        summary = {"checked": len(entries), "with_problems": sum(1 for e in entries if e["syntax_errors"] or e["problems"] or e["issues"]),
                   "fixed": len(fixed), "skipped": len(skipped), "ignored": ignored,
                   "lines": sum(e["lines"] for e in entries), "ai": use_ai, **stats}
        job["fixed"] = fixed
        job["result"] = {"success": True, "mode": mode, "files": entries, "skipped": skipped,
                         "summary": summary, "notes": notes, "cancelled": cancelled}


def _cp_thread(ui, job, uploads, ai):
    try:
        _cp_run(ui, job, uploads, ai)
    except Exception as e:
        print("Code project error:", e)
        job["result"] = {"success": False, "error": "Something went wrong: " + str(e)[:200]}
    finally:
        try:
            ui.code_checker._cancel_check = None
        except Exception:
            pass
    job["pct"], job["stage"], job["done"] = 100, "Done", True


def _cp_add_routes(self):
    from flask import Response
    app = self.app
    app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024
    app.config["MAX_FORM_PARTS"] = 5000
    app.config["MAX_FORM_MEMORY_SIZE"] = 20 * 1024 * 1024

    @app.route("/api/check_files_start", methods=["POST"])
    def check_files_start():
        fl = request.files.getlist("files")
        if not fl:
            return jsonify({"success": False, "error": "No files received."}), 400
        try:
            paths = json.loads(request.form.get("paths_json") or "[]")
        except Exception:
            paths = []
        uploads = []
        for i, f in enumerate(fl):
            name = paths[i] if i < len(paths) and paths[i] else (f.filename or f"file{i}")
            uploads.append((str(name), f.read()))
        ai = request.form.get("fast") != "1"
        mode = (request.form.get("mode") or "fix").strip().lower()
        if mode not in ("fix", "improve"):
            mode = "fix"
        now = time.time()
        for k in [k for k, v in CODE_JOBS.items() if now - v["t"] > 3600]:
            CODE_JOBS.pop(k, None)
        jid = uuid.uuid4().hex
        job = {"t": now, "done": False, "cancel": False, "stage": "Queued", "file": "", "pct": 1,
               "result": None, "fixed": {}, "mode": mode}
        CODE_JOBS[jid] = job
        threading.Thread(target=_cp_thread, args=(self, job, uploads, ai), daemon=True).start()
        return jsonify({"success": True, "job": jid})

    @app.route("/api/code_status/<jid>")
    def code_status(jid):
        j = CODE_JOBS.get(jid)
        if not j:
            return jsonify({"success": False, "error": "Unknown or expired job."}), 404
        out = {"success": True, "done": j["done"], "stage": j["stage"], "pct": j["pct"], "file": j["file"]}
        if j["done"]:
            out["result"] = j["result"]
        return jsonify(out)

    @app.route("/api/code_cancel/<jid>", methods=["POST"])
    def code_cancel(jid):
        j = CODE_JOBS.get(jid)
        if j:
            j["cancel"] = True
        return jsonify({"success": True})

    @app.route("/api/code_fixed/<jid>")
    def code_fixed(jid):
        j = CODE_JOBS.get(jid)
        code = j["fixed"].get(request.args.get("path", "")) if j else None
        if code is None:
            return "Not found.", 404
        return Response(code, mimetype="text/plain; charset=utf-8")

    @app.route("/api/code_download/<jid>")
    def code_download(jid):
        j = CODE_JOBS.get(jid)
        if not j or not j["done"] or not j["result"] or not j["result"].get("success"):
            return "Not found.", 404
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p, c in j["fixed"].items():
                z.writestr(p, c)
            z.writestr("REPORT.md", _cp_report_md(j["result"]))
        return Response(buf.getvalue(), mimetype="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="pip-ai-fixed-code.zip"'})

    @app.route("/api/code_report/<jid>")
    def code_report(jid):
        j = CODE_JOBS.get(jid)
        if not j or not j["done"] or not j["result"] or not j["result"].get("success"):
            return "Not found.", 404
        return Response(_cp_report_md(j["result"]), mimetype="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="pip-ai-code-report.md"'})


_cp_prev_init = WebUI.__init__


def _cp_init(self):
    _cp_prev_init(self)
    if FLASK_AVAILABLE:
        _cp_add_routes(self)


WebUI.__init__ = _cp_init

# ---------------------------------------------------------------- page
CP_HTML = r"""
<div class="cu-box" id="cu-drop">
  <div class="cu-t">📁 Or upload code: files, a whole folder, or a ZIP</div>
  <div class="oc-row" style="justify-content:center">
    <button type="button" class="oc-btn" id="cu-files">Choose files</button>
    <button type="button" class="oc-btn" id="cu-folder">Choose folder</button>
  </div>
  <div class="cu-sub">…or drag and drop them here. Works with .zip, .tar, .tar.gz and many languages (Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, PHP, Ruby, Kotlin, Swift…). Big files are checked part by part.</div>
  <input type="file" id="cu-in-files" multiple hidden>
  <input type="file" id="cu-in-folder" webkitdirectory directory multiple hidden>
  <div id="cu-list"></div>
</div>
"""

# shown right above the "Check Code" button so the choice is impossible to miss
CP_MODE_HTML = r"""
<div class="cu-opts">
  <label class="mx-chk"><input type="radio" name="cu-mode" value="fix" checked> 🔧 Fix errors — what's wrong, why, examples + the full corrected code + other right ways</label>
  <label class="mx-chk"><input type="radio" name="cu-mode" value="improve"> ✨ Improve — analysis, tips with examples, other right ways + the full improved code + why it's better</label>
  <label class="mx-chk"><input type="checkbox" id="cu-fast"> ⚡ Fast scan (no AI, only syntax and safety rules: good for huge projects)</label>
</div>
"""

CP_CSS = r"""
.cu-box{margin-top:12px;padding:16px;border:2px dashed rgba(221,132,72,.7);border-radius:16px;background:rgba(0,0,0,.25);text-align:center;transition:all .2s}
.cu-box.over{background:rgba(221,132,72,.25);border-color:#ffaa5c}
.cu-t{font-weight:700;margin-bottom:10px}
.cu-sub{font-size:.85em;opacity:.75;margin-top:8px}
.cu-opts{margin-top:12px;display:flex;flex-direction:column;gap:6px;align-items:flex-start}
.cu-opts .mx-chk{margin:0}
#cu-list{text-align:left;margin-top:10px}
.cu-sum{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px;font-size:.92em}
.cu-item{display:flex;gap:10px;align-items:center;padding:4px 8px;border-radius:8px;font-size:.88em}
.cu-item:hover{background:rgba(255,255,255,.07)}
.cu-p{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'Courier New',monospace}
.cu-item small{opacity:.7}
.cu-item button{background:none;border:0;cursor:pointer;color:inherit}
.cf{margin:8px 0;border-radius:12px;border:1px solid rgba(255,255,255,.2);background:rgba(0,0,0,.2)}
.cf summary{cursor:pointer;padding:10px 14px;font-weight:600;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.cf summary small{font-weight:400;opacity:.8}
.cf .cf-body{padding:0 14px 12px}
.cf-path{font-family:'Courier New',monospace;word-break:break-all}
.cf pre{max-height:420px;overflow:auto}
.cf.bad{border-color:rgba(255,170,92,.7)}
body.light .cu-box{background:rgba(255,255,255,.4)}
body.light .cf{background:rgba(255,255,255,.5);border-color:rgba(60,120,200,.3)}

/* ===== mini games while the AI checks ===== */
:root{--gm-board:#0d1526;--gm-grid:rgba(255,255,255,.07);--gm-ink:#e8eefc;--gm-accent:#60a5fa}
body.light{--gm-board:#eaf4ff;--gm-grid:rgba(30,80,150,.14);--gm-ink:#12304f;--gm-accent:#2563eb}
.gm-pick,.gm-wrap{margin-top:16px}
.gm-pt{font-weight:700;margin-bottom:10px;text-align:center}
.gm-cards{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
.gm-card{display:flex;flex-direction:column;align-items:center;gap:5px;min-width:180px;padding:18px 22px;border-radius:18px;cursor:pointer;
  border:2px solid rgba(221,132,72,.6);background:rgba(255,255,255,.07);color:inherit;font-family:inherit;transition:all .2s}
.gm-card:hover{background:rgba(221,132,72,.28);transform:translateY(-3px);box-shadow:0 10px 24px rgba(221,132,72,.3)}
.gm-ico{font-size:36px}.gm-card b{font-size:18px}.gm-card small{opacity:.75;font-size:12.5px;text-align:center;max-width:210px}
.gm-bar{display:flex;gap:10px;align-items:center;justify-content:center;margin-bottom:10px;flex-wrap:wrap}
.gm-bar>b{font-size:17px}
.gm-score{font-weight:800;color:var(--gm-accent);min-width:64px;text-align:center}
.gm-cv{display:block;margin:0 auto;max-width:100%;height:auto;border-radius:12px;border:2px solid var(--gm-accent);background:var(--gm-board);touch-action:none}
.gm-hint{margin-top:8px;text-align:center;font-size:.85em;opacity:.75}
body.light .gm-card{background:rgba(255,255,255,.55);border-color:rgba(60,120,200,.5)}
body.light .gm-card:hover{background:rgba(255,200,220,.65);box-shadow:0 10px 24px rgba(255,122,162,.3)}
"""

CP_JS = r"""
(function () {
  const $ = id => document.getElementById(id);
  const ta = $('code-input'), out = $('code-output'), drop = $('cu-drop');
  if (!ta || !out || !drop) return;
  const esc = s => escapeHtml(String(s == null ? '' : s));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const EXTS = new Set(__EXTS__);
  const SKIP = new Set(['.git','node_modules','__pycache__','venv','.venv','env','dist','build','.idea','.vscode','site-packages','target','vendor','.next','__macosx','.mypy_cache','.pytest_cache','.gradle','.tox']);
  const ARCH = /\.(zip|tar|tgz|tar\.gz|tar\.bz2|tar\.xz)$/i;
  const MAXF = 8 * 1048576, MAXA = 95 * 1048576;
  let picked = [], ignored = 0, tooBig = [];
  const fmt = n => n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(0) + ' KB' : (n / 1048576).toFixed(1) + ' MB';
  const list = a => '<ul>' + a.map(x => '<li>' + esc(x) + '</li>').join('') + '</ul>';
  const CODE_CSS = 'display:block;margin-top:5px;padding:7px 9px;background:rgba(127,127,127,.18);border-radius:8px;white-space:pre-wrap;font-size:.92em';
  function bulletHtml(t) {
    t = String(t);
    const marks = [];
    const re = /(Before:|Example:)/g;
    let m;
    while ((m = re.exec(t))) marks.push({ i: m.index, label: m[1] });
    if (!marks.length) return '<li>' + esc(t) + '</li>';
    let h = '<li>' + esc(t.slice(0, marks[0].i));
    for (let k = 0; k < marks.length; k++) {
      const end = k + 1 < marks.length ? marks[k + 1].i : t.length;
      const code = t.slice(marks[k].i + marks[k].label.length, end).trim();
      h += '<br><b>' + marks[k].label + '</b><code style="' + CODE_CSS + '">' + esc(code) + '</code>';
    }
    return h + '</li>';
  }
  const bullets = a => '<ul>' + a.map(bulletHtml).join('') + '</ul>';

  function accept(path, size) {
    const parts = path.split('/'), name = parts[parts.length - 1].toLowerCase();
    if (parts.slice(0, -1).some(p => SKIP.has(p.toLowerCase()))) return false;
    if (ARCH.test(name)) { if (size > MAXA) { tooBig.push(path); return false; } return true; }
    const m = name.match(/\.[^.]+$/), ext = m ? m[0] : '';
    if (ext && !EXTS.has(ext)) return false;
    if (!ext && size > 200000) return false;
    if (size > MAXF) { tooBig.push(path); return false; }
    return true;
  }
  function addItems(arr) {
    arr.forEach(x => {
      if (!accept(x.path, x.file.size)) { ignored++; return; }
      const i = picked.findIndex(p => p.path === x.path);
      if (i >= 0) picked[i] = x; else picked.push(x);
    });
    render();
  }
  function render() {
    const box = $('cu-list');
    if (!picked.length && !ignored && !tooBig.length) { box.innerHTML = ''; return; }
    const total = picked.reduce((a, x) => a + x.file.size, 0);
    let h = '<div class="cu-sum"><span>' + picked.length + ' file(s) · ' + fmt(total) + (ignored ? ' · ' + ignored + ' non-code file(s) ignored' : '') +
            '</span><button type="button" class="oc-btn" data-act="clear">🗑 Clear</button></div>';
    if (tooBig.length) h += '<div class="ppl-note">Too large to upload (8 MB per code file, 95 MB per archive): ' + esc(tooBig.slice(0, 5).join(', ')) + (tooBig.length > 5 ? '…' : '') + '</div>';
    h += picked.slice(0, 40).map((x, i) => '<div class="cu-item"><span class="cu-p">' + esc(x.path) + '</span><small>' + fmt(x.file.size) +
         '</small><button type="button" data-act="rm" data-i="' + i + '">✖</button></div>').join('');
    if (picked.length > 40) h += '<div class="ppl-note">… and ' + (picked.length - 40) + ' more</div>';
    box.innerHTML = h;
  }
  $('cu-list').addEventListener('click', e => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    if (b.dataset.act === 'clear') { picked = []; ignored = 0; tooBig = []; }
    else picked.splice(+b.dataset.i, 1);
    render();
  });

  $('cu-files').addEventListener('click', () => $('cu-in-files').click());
  $('cu-folder').addEventListener('click', () => $('cu-in-folder').click());
  [['cu-in-files'], ['cu-in-folder']].forEach(([id]) => {
    const el = $(id);
    el.addEventListener('change', () => {
      addItems(Array.from(el.files).map(f => ({ file: f, path: f.webkitRelativePath || f.name })));
      el.value = '';
    });
  });

  async function walk(entry, base, acc) {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      acc.push({ file: f, path: base + entry.name });
    } else if (entry.isDirectory) {
      if (SKIP.has(entry.name.toLowerCase())) return;
      const r = entry.createReader();
      let all = [];
      while (true) {
        const b = await new Promise((res, rej) => r.readEntries(res, rej));
        if (!b.length) break;
        all = all.concat(b);
      }
      for (const e of all) await walk(e, base + entry.name + '/', acc);
    }
  }
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  drop.addEventListener('dragleave', e => { e.preventDefault(); drop.classList.remove('over'); });
  drop.addEventListener('drop', async e => {
    e.preventDefault(); drop.classList.remove('over');
    const dt = e.dataTransfer, entries = [];
    if (dt.items) for (const it of dt.items) { const en = it.webkitGetAsEntry ? it.webkitGetAsEntry() : null; if (en) entries.push(en); }
    if (entries.length) { const acc = []; for (const en of entries) await walk(en, '', acc); addItems(acc); }
    else addItems(Array.from(dt.files).map(f => ({ file: f, path: f.name })));
  });

  // ---------- upload + progress ----------
  function upload(items, fast, mode, onProg, setX) {
    return new Promise((res, rej) => {
      const fd = new FormData();
      items.forEach(it => fd.append('files', it.file, it.path.split('/').pop() || 'file'));
      fd.append('paths_json', JSON.stringify(items.map(i => i.path)));
      fd.append('fast', fast ? '1' : '0');
      fd.append('mode', mode || 'fix');
      const x = new XMLHttpRequest(); setX(x);
      x.open('POST', '/api/check_files_start');
      x.upload.onprogress = e => { if (e.lengthComputable) onProg(e.loaded / e.total); };
      x.onload = () => { try { res(JSON.parse(x.responseText)); } catch (e) { rej(new Error('The server answered with a web page (HTTP ' + x.status + '). The upload may be too big for the tunnel (about 100 MB). Try fewer files.')); } };
      x.onerror = () => rej(new Error('The upload failed.'));
      x.onabort = () => rej(new Error('Stopped.'));
      x.send(fd);
    });
  }

  const ST = {
    correct: ['ok', '✅ Correct'], fixed: ['ok', '🔧 Fixed'], rule_based_fixed: ['ok', '🔧 Basic fixes'],
    fixed_unverified: ['warn', '⚠️ Fixed, not fully verified'], suggestion_only: ['warn', '⚠️ Problems found'],
    tips: ['ok', '💡 Tips'], improved: ['ok', '✨ Improved'], clean: ['ok', '✅ Nothing to improve'],
    no_model: ['warn', '⚠️ Basic check only'], scanned: ['ok', '✅ Nothing found']
  };
  let R = null, JID = null;

  function fileHtml(f, i, open) {
    const st = ST[f.status] || ['warn', f.status];
    const bad = f.syntax_errors.length + f.problems.length + f.issues.length;
    let h = '<details class="cf' + (bad ? ' bad' : '') + '"' + (open ? ' open' : '') + '><summary>' + st[1] +
            ' <span class="cf-path">' + esc(f.path) + '</span><small>' + f.language + ' · ' + f.lines.toLocaleString() + ' lines' +
            (bad ? ' · ' + (f.problems_total || bad) + ' finding(s)' : '') + '</small></summary><div class="cf-body">';
    if (f.syntax_errors.length) h += '<div class="cr-title">⚠️ Parser report</div>' + list(f.syntax_errors);
    if (f.problems.length) h += '<div class="cr-title">❌ What is wrong, why it fails, and the right way</div>' + bullets(f.problems) +
        (f.problems_total > f.problems.length ? '<p><small>… and ' + (f.problems_total - f.problems.length) + ' more.</small></p>' : '');
    else if (f.issues.length) h += '<div class="cr-title">🔎 Safety / quality checks</div>' + bullets(f.issues);
    if (f.analysis && f.analysis.length) h += '<div class="cr-title">🔎 Analysis</div>' + list(f.analysis);
    if (f.tips && f.tips.length) h += '<div class="cr-title">💡 Improvement tips — why it is better, before vs after</div>' + bullets(f.tips);
    if (f.other_ways && f.other_ways.length) h += '<div class="cr-title">🔀 Other right ways to do it</div>' + bullets(f.other_ways);
    if (f.changed) {
      h += '<div class="cr-title">' + (f.kind === 'improve' ? '✨ Improved code' : '✅ Corrected code') + '</div><div class="oc-row" style="margin:0 0 6px">' +
           '<button type="button" class="oc-btn" data-act="show" data-i="' + i + '">👁 Show</button>' +
           '<button type="button" class="oc-btn" data-act="copy" data-i="' + i + '">📋 Copy</button>' +
           '<button type="button" class="oc-btn" data-act="dl" data-i="' + i + '">⬇ Download</button></div><div class="cf-code" id="cf-code-' + i + '"></div>';
    }
    if (f.why_better && f.why_better.length) h += '<div class="cr-title">💡 Why the improved code is better</div>' + bullets(f.why_better);
    if (f.why_correct.length) h += '<div class="cr-title">💡 Why this is correct</div>' + bullets(f.why_correct);
    if (f.notes.length) h += '<div class="cr-title">📝 Notes</div>' + list(f.notes);
    if (!bad && !f.changed && !f.notes.length && !(f.tips && f.tips.length)) h += '<p>Nothing to report for this file.</p>';
    return h + '</div></details>';
  }
  function renderResult(res, jid) {
    if (!res.success) return '<div class="cr-status warn">⚠️ ' + esc(res.error || 'Something went wrong.') + '</div>';
    R = res; JID = jid;
    const s = res.summary;
    const isImp = res.mode === 'improve';
    let h = '<div class="cr-status ' + (s.with_problems ? 'warn' : 'ok') + '">' +
            (isImp
              ? (s.fixed ? '✨ Improvements suggested for ' + s.fixed + ' of ' + s.checked + ' file(s)' : '✅ Nothing to improve in ' + s.checked + ' file(s)')
              : (s.with_problems ? '⚠️ Found problems in ' + s.with_problems + ' of ' + s.checked + ' file(s)' : '✅ No problems found in ' + s.checked + ' file(s)')) +
            (res.cancelled ? ' (stopped early)' : '') + '</div>';
    const chips = [s.lines.toLocaleString() + ' lines', (isImp ? s.fixed + ' improved' : s.fixed + ' fixed'), s.skipped + ' skipped', s.ignored + ' non-code ignored'];
    if (s.ai) chips.push('AI reviewed ' + s.parts_reviewed + '/' + s.parts_total + ' parts');
    h += '<div class="ppl-chips">' + chips.map(c => '<span class="ppl-chip">' + esc(c) + '</span>').join('') + '</div>';
    h += '<div class="oc-row">' + (s.fixed ? '<button type="button" class="oc-btn" data-act="zip">⬇ ' + (isImp ? 'Improved' : 'Fixed') + ' project (ZIP)</button>' : '') +
         '<button type="button" class="oc-btn" data-act="report">⬇ Report (.md)</button></div>';
    if (res.notes && res.notes.length) h += '<div class="cr-title">📝 Notes</div>' + list(res.notes);
    h += '<div class="cr-title">📄 Files</div>' + res.files.map((f, i) => fileHtml(f, i, res.files.length === 1)).join('');
    if (res.skipped && res.skipped.length) h += '<details class="cf"><summary>Skipped files (' + res.skipped.length + ')</summary><div class="cf-body">' +
        list(res.skipped.slice(0, 200).map(x => x.path + ': ' + x.reason)) + '</div></details>';
    return h;
  }
  const codeCache = {};
  async function getCode(i) {
    const f = R.files[i];
    if (codeCache[JID + f.path] == null) {
      const r = await fetch('/api/code_fixed/' + JID + '?path=' + encodeURIComponent(f.path));
      codeCache[JID + f.path] = r.ok ? await r.text() : '';
    }
    return codeCache[JID + f.path];
  }
  out.addEventListener('click', async e => {
    const b = e.target.closest('[data-act]'); if (!b || !R) return;
    const a = b.dataset.act;
    if (a === 'zip') { location.href = '/api/code_download/' + JID; return; }
    if (a === 'report') { location.href = '/api/code_report/' + JID; return; }
    const i = +b.dataset.i, f = R.files[i];
    const code = await getCode(i);
    if (a === 'show') {
      const box = $('cf-code-' + i);
      box.innerHTML = '<pre><code>' + esc(code.length > 200000 ? code.slice(0, 200000) + '\n… (shortened here, download it to get everything)' : code) + '</code></pre>';
    } else if (a === 'copy') {
      navigator.clipboard.writeText(code).then(() => { const o = b.textContent; b.textContent = 'Copied ✓'; setTimeout(() => { b.textContent = o; }, 1500); });
    } else if (a === 'dl') {
      const blob = new Blob([code], { type: 'text/plain' }), l = document.createElement('a');
      l.href = URL.createObjectURL(blob); l.download = 'fixed_' + f.path.split('/').pop();
      document.body.appendChild(l); l.click(); l.remove(); setTimeout(() => URL.revokeObjectURL(l.href), 3000);
    }
  });

  // ---------- mini games while the AI checks ----------
  const GM = { tick: null, raf: 0, kind: '', keys: null, restart: null };
  function gmColors() {
    const cs = getComputedStyle(document.body);
    const v = n => (cs.getPropertyValue(n) || '').trim();
    return { board: v('--gm-board') || '#0d1526', grid: v('--gm-grid') || 'rgba(255,255,255,.07)',
             ink: v('--gm-ink') || '#e8eefc', accent: v('--gm-accent') || '#60a5fa',
             snake: v('--gm-snake') || '#4ade80', head: v('--gm-snakehead') || '#d1fae5',
             food: v('--gm-food') || '#fb7185' };
  }
  function gmStop() {
    if (GM.tick) { clearInterval(GM.tick); GM.tick = null; }
    if (GM.raf) { cancelAnimationFrame(GM.raf); GM.raf = 0; }
    GM.restart = null; GM.keys = null; GM.kind = '';
  }
  function gmPicker() {
    gmStop();
    const g = $('cu-games'); if (!g) return;
    g.innerHTML = '<div class="gm-pick"><div class="gm-pt">⏳ Still checking — play a mini game while you wait</div>' +
      '<div class="gm-cards">' +
      '<button type="button" class="gm-card" data-gm="snake"><span class="gm-ico">🐍</span><b>Snake</b><small>Eat the dots, grow long, don\'t bite yourself</small></button>' +
      '<button type="button" class="gm-card" data-gm="blocks"><span class="gm-ico">🧱</span><b>Block Stack</b><small>Catch falling blocks, clear lines, don\'t reach the top</small></button>' +
      '</div></div>';
  }
  function gmShell(kind, title, hint) {
    gmStop(); GM.kind = kind;
    const g = $('cu-games'); if (!g) return null;
    g.innerHTML = '<div class="gm-wrap"><div class="gm-bar"><button type="button" class="oc-btn" data-gm-back>← Games</button>' +
      '<b>' + title + '</b><span class="gm-score" id="gm-score">0</span>' +
      '<button type="button" class="oc-btn" data-gm-restart>↻ Restart</button></div>' +
      '<canvas id="gm-cv" class="gm-cv"></canvas><div class="gm-hint">' + hint + '</div></div>';
    return $('gm-cv');
  }
  out.addEventListener('click', e => {
    const gm = e.target.closest('[data-gm]'), bk = e.target.closest('[data-gm-back]'), rs = e.target.closest('[data-gm-restart]');
    if (gm) { if (gm.dataset.gm === 'snake') startSnake(); else if (gm.dataset.gm === 'blocks') startBlocks(); return; }
    if (bk) { gmPicker(); return; }
    if (rs && GM.restart) GM.restart();
  });
  window.addEventListener('keydown', e => {
    if (!GM.kind || !GM.keys) return;
    const t = document.activeElement;
    if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
    GM.keys(e);
  });

  function startSnake() {
    const cv = gmShell('snake', '🐍 Snake', 'Arrow keys / WASD to move · R or ↻ to restart');
    if (!cv) return;
    const C = gmColors(), COLS = 24, ROWS = 18, CELL = 20, W = COLS * CELL, H = ROWS * CELL;
    cv.width = W; cv.height = H;
    let snake, dir, nd, food, score, dead;
    const scoreEl = () => { const s = $('gm-score'); if (s) s.textContent = score; };
    function placeFood() {
      let f; do { f = { x: (Math.random() * COLS) | 0, y: (Math.random() * ROWS) | 0 }; }
      while (snake.some(s => s.x === f.x && s.y === f.y));
      food = f;
    }
    function draw() {
      const x = cv.getContext('2d');
      x.fillStyle = C.board; x.fillRect(0, 0, W, H);
      x.strokeStyle = C.grid; x.lineWidth = 1;
      for (let i = 1; i < COLS; i++) { x.beginPath(); x.moveTo(i * CELL, 0); x.lineTo(i * CELL, H); x.stroke(); }
      for (let j = 1; j < ROWS; j++) { x.beginPath(); x.moveTo(0, j * CELL); x.lineTo(W, j * CELL); x.stroke(); }
      x.fillStyle = C.food; x.beginPath(); x.arc(food.x * CELL + CELL / 2, food.y * CELL + CELL / 2, CELL * .36, 0, 7); x.fill();
      snake.forEach((s, i) => { x.fillStyle = i ? C.snake : C.head; x.fillRect(s.x * CELL + 1, s.y * CELL + 1, CELL - 2, CELL - 2); });
      if (dead) {
        x.fillStyle = 'rgba(0,0,0,.5)'; x.fillRect(0, 0, W, H);
        x.fillStyle = '#fff'; x.textAlign = 'center'; x.font = 'bold 24px sans-serif';
        x.fillText('Game over — score ' + score, W / 2, H / 2 - 8);
        x.font = '15px sans-serif'; x.fillText('Press R or hit ↻ Restart', W / 2, H / 2 + 24); x.textAlign = 'left';
      }
    }
    function reset() {
      snake = [{ x: 12, y: 9 }, { x: 11, y: 9 }, { x: 10, y: 9 }];
      dir = { x: 1, y: 0 }; nd = { x: 1, y: 0 }; score = 0; dead = false;
      placeFood(); scoreEl(); draw();
      if (GM.tick) clearInterval(GM.tick);
      GM.tick = setInterval(tick, 120);
    }
    function tick() {
      if (!cv.isConnected) { gmStop(); return; }
      dir = nd;
      const h = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
      if (h.x < 0 || h.y < 0 || h.x >= COLS || h.y >= ROWS || snake.some(s => s.x === h.x && s.y === h.y)) {
        dead = true; clearInterval(GM.tick); GM.tick = null; draw(); return;
      }
      snake.unshift(h);
      if (h.x === food.x && h.y === food.y) { score++; scoreEl(); placeFood(); }
      else snake.pop();
      draw();
    }
    GM.keys = e => {
      const k = e.key.toLowerCase();
      const m = { arrowup: [0, -1], w: [0, -1], arrowdown: [0, 1], s: [0, 1], arrowleft: [-1, 0], a: [-1, 0], arrowright: [1, 0], d: [1, 0] };
      if (m[k]) { e.preventDefault(); const d = m[k]; if (d[0] !== -dir.x || d[1] !== -dir.y) nd = { x: d[0], y: d[1] }; }
      else if ((k === 'r' || k === 'enter') && dead) reset();
    };
    GM.restart = reset;
    reset();
  }

  function startBlocks() {
    const cv = gmShell('blocks', '🧱 Block Stack', '← → move · ↑ / X rotate · ↓ soft drop · Space hard drop');
    if (!cv) return;
    const C = gmColors(), COLS = 10, ROWS = 20, CELL = 30, W = COLS * CELL, H = ROWS * CELL;
    cv.width = W; cv.height = H;
    const SHAPES = [
      [[1, 1, 1, 1]], [[1, 1], [1, 1]], [[0, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 1, 1]],
      [[0, 0, 1], [1, 1, 1]], [[0, 1, 1], [1, 1, 0]], [[1, 1, 0], [0, 1, 1]]];
    const PCOL = ['#22d3ee', '#facc15', '#c084fc', '#4ade80', '#fb7185', '#60a5fa', '#fb923c'];
    let board, cur, nxt, score, lines, level, dead, dropN, grav;
    const scoreEl = () => { const s = $('gm-score'); if (s) s.textContent = score; };
    const rot = m => m[0].map((_, i) => m.map(r => r[r.length - 1 - i]));
    function mk() { const t = (Math.random() * 7) | 0; return { m: SHAPES[t].map(r => r.slice()), c: PCOL[t], x: 3, y: 0 }; }
    function collide(m, px, py) {
      for (let y = 0; y < m.length; y++) for (let x = 0; x < m[y].length; x++) {
        if (!m[y][x]) continue;
        const bx = px + x, by = py + y;
        if (bx < 0 || bx >= COLS || by >= ROWS) return true;
        if (by >= 0 && board[by][bx]) return true;
      }
      return false;
    }
    function lock() {
      cur.m.forEach((r, y) => r.forEach((v, x) => { if (v && cur.y + y >= 0) board[cur.y + y][cur.x + x] = cur.c; }));
      let cleared = 0;
      for (let y = ROWS - 1; y >= 0; y--) {
        if (board[y].every(v => v)) { board.splice(y, 1); board.unshift(Array(COLS).fill(0)); cleared++; y++; }
      }
      if (cleared) {
        lines += cleared; score += [0, 100, 300, 500, 800][cleared] * level;
        level = 1 + ((lines / 10) | 0); grav = Math.max(90, 750 - (level - 1) * 70); scoreEl();
      }
      cur = nxt; nxt = mk();
      if (collide(cur.m, cur.x, cur.y)) { dead = true; clearInterval(GM.tick); GM.tick = null; draw(); }
    }
    function step() { if (collide(cur.m, cur.x, cur.y + 1)) lock(); else { cur.y++; draw(); } }
    function hard() { while (!collide(cur.m, cur.x, cur.y + 1)) cur.y++; lock(); }
    function reset() {
      board = Array.from({ length: ROWS }, () => Array(COLS).fill(0));
      score = 0; lines = 0; level = 1; grav = 750; dead = false; dropN = 0;
      cur = mk(); nxt = mk(); scoreEl(); draw();
      if (GM.tick) clearInterval(GM.tick);
      GM.tick = setInterval(() => {
        if (!cv.isConnected) { gmStop(); return; }
        if (++dropN >= Math.round(grav / 50)) { dropN = 0; if (!dead) step(); }
      }, 50);
    }
    function draw() {
      const x = cv.getContext('2d');
      x.fillStyle = C.board; x.fillRect(0, 0, W, H);
      x.strokeStyle = C.grid; x.lineWidth = 1;
      for (let i = 1; i < COLS; i++) { x.beginPath(); x.moveTo(i * CELL, 0); x.lineTo(i * CELL, H); x.stroke(); }
      for (let j = 1; j < ROWS; j++) { x.beginPath(); x.moveTo(0, j * CELL); x.lineTo(W, j * CELL); x.stroke(); }
      for (let y = 0; y < ROWS; y++) for (let i = 0; i < COLS; i++) if (board[y][i]) {
        x.fillStyle = board[y][i]; x.fillRect(i * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2);
      }
      if (cur && !dead) cur.m.forEach((r, y) => r.forEach((v, i2) => {
        if (v && cur.y + y >= 0) { x.fillStyle = cur.c; x.fillRect((cur.x + i2) * CELL + 1, (cur.y + y) * CELL + 1, CELL - 2, CELL - 2); }
      }));
      x.fillStyle = C.ink; x.font = '13px sans-serif'; x.fillText('Next', W - 72, 20);
      nxt.m.forEach((r, y) => r.forEach((v, i2) => {
        if (v) { x.fillStyle = nxt.c; x.fillRect(W - 66 + i2 * 16, 30 + y * 16, 14, 14); }
      }));
      if (dead) {
        x.fillStyle = 'rgba(0,0,0,.55)'; x.fillRect(0, 0, W, H);
        x.fillStyle = '#fff'; x.textAlign = 'center'; x.font = 'bold 24px sans-serif';
        x.fillText('Game over — ' + score, W / 2, H / 2 - 8);
        x.font = '15px sans-serif'; x.fillText('Press R or hit ↻ Restart', W / 2, H / 2 + 24); x.textAlign = 'left';
      }
    }
    GM.keys = e => {
      const k = e.key.toLowerCase();
      if (dead) { if (k === 'r' || k === 'enter') { e.preventDefault(); reset(); } return; }
      if (k === 'arrowleft' || k === 'a') { e.preventDefault(); if (!collide(cur.m, cur.x - 1, cur.y)) { cur.x--; draw(); } }
      else if (k === 'arrowright' || k === 'd') { e.preventDefault(); if (!collide(cur.m, cur.x + 1, cur.y)) { cur.x++; draw(); } }
      else if (k === 'arrowdown' || k === 's') { e.preventDefault(); if (!collide(cur.m, cur.x, cur.y + 1)) { cur.y++; score++; scoreEl(); draw(); } }
      else if (k === 'arrowup' || k === 'x' || k === 'w') { e.preventDefault(); const r = rot(cur.m); if (!collide(r, cur.x, cur.y)) { cur.m = r; draw(); } }
      else if (k === ' ') { e.preventDefault(); hard(); }
    };
    GM.restart = reset;
    reset();
  }

  // ---------- main button ----------
  let busy = false;
  window.checkCode = async function () {
    if (busy) return;
    const items = picked.slice(), code = ta.value;
    if (code.trim()) {
      const ext = { python: '.py', javascript: '.js', java: '.java', cpp: '.cpp' }[$('code-language').value] || '.txt';
      items.push({ file: new Blob([code], { type: 'text/plain' }), path: 'pasted_code' + ext });
    }
    if (!items.length) { out.textContent = 'Please paste some code or upload files.'; return; }
    const total = items.reduce((a, x) => a + (x.file.size || 0), 0);
    if (total > MAXA) { out.textContent = 'Everything together is ' + fmt(total) + '. The limit per check is about 95 MB. Upload fewer files.'; return; }
    busy = true;
    let jid = null, curX = null, stopped = false, target = 1, shown = 0, done = false;
    out.innerHTML = '<div class="an-stage"><span id="cu-stage">Uploading</span><span class="an-dots"><i></i><i></i><i></i></span></div>' +
      '<div class="an-bar"><i id="cu-fill"></i></div><div class="an-meta"><span id="cu-pct">0%</span><span id="cu-time">0s</span></div>' +
      '<div class="an-hint" id="cu-file"></div><div class="oc-row"><button type="button" class="oc-btn" id="cu-stop">⏹ Stop</button></div>' +
      '<div id="cu-games"></div>';
    out.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    const t0 = Date.now();
    const timer = setInterval(() => {
      shown += (target - shown) * 0.12;
      const f = $('cu-fill'); if (!f) return;
      f.style.width = shown + '%'; $('cu-pct').textContent = Math.round(shown) + '%';
      $('cu-time').textContent = Math.floor((Date.now() - t0) / 1000) + 's';
    }, 100);
    $('cu-stop').addEventListener('click', () => {
      stopped = true;
      $('cu-stop').disabled = true;
      if (!jid) { if (curX) curX.abort(); return; }
      $('cu-stage').textContent = 'Stopping…';
      fetch('/api/code_cancel/' + jid, { method: 'POST' });
    });
    gmPicker();
    try {
      const mode = (document.querySelector('input[name="cu-mode"]:checked') || {}).value || 'fix';
      const start = await upload(items, $('cu-fast').checked, mode, p => { target = Math.max(target, p * 12); $('cu-stage').textContent = 'Uploading ' + Math.round(p * 100) + '%'; }, x => { curX = x; });
      if (!start.success) throw new Error(start.error || 'Could not start.');
      jid = start.job;
      let res = null, fails = 0;
      while (!res) {
        await sleep(400);
        try {
          const s = JSON.parse(await (await fetch('/api/code_status/' + jid)).text());
          fails = 0;
          if (!s.success) throw new Error(s.error);
          target = Math.max(target, s.pct);
          if (!stopped) $('cu-stage').textContent = s.stage;
          $('cu-file').textContent = s.file || '';
          if (s.done) res = s.result;
        } catch (e) { if (++fails > 60) throw e; }
      }
      done = true; target = 100; await sleep(300);
      out.innerHTML = renderResult(res, jid);
    } catch (e) {
      out.innerHTML = '<div class="cr-status warn">⚠️ ' + esc(e.message) + '</div>';
    } finally {
      clearInterval(timer); busy = false;
    }
  };
})();
"""

_cp_prev_get_html = WebUI.get_html


def _cp_get_html(self) -> str:
    html = _cp_prev_get_html(self)
    html = re.sub(r'(<textarea id="code-input"[^>]*></textarea>)', lambda m: m.group(1) + CP_HTML, html, count=1)
    html = html.replace('<button class="generate-button" onclick="checkCode()"><span>Check Code</span></button>',
                        CP_MODE_HTML + '<button class="generate-button" onclick="checkCode()"><span>Check Code</span></button>', 1)
    html = html.replace("<label>Your Code</label>", "<label>Your Code (paste it here, or upload files below)</label>", 1)
    html = html.replace("</head>", "<style>" + CP_CSS + "</style></head>", 1)
    js = CP_JS.replace("__EXTS__", json.dumps(sorted(set(_CP_EXT))))
    return html.replace("</body>", "<script>" + js + "</script></body>", 1)


WebUI.get_html = _cp_get_html

# ============================================================================
# NEWS VOICE + LANGUAGE: a mic button and a searchable language picker on the
# Fake News tab. Voice is transcribed (Whisper, many languages), the transcript
# is checked automatically, and 🔊 reads the verdict aloud in that language.
# The image-analysis agent chat gets the same mic button.
# ============================================================================
def _news_add_routes(self):
    @self.app.route('/api/stt_start', methods=['POST'])
    def stt_start():
        """Speech-to-text only (no chat reply) - used by Fake News and image agent."""
        f = request.files.get('audio')
        if not f:
            return jsonify({'success': False, 'error': 'No audio received.'}), 400
        lang_pref = (request.form.get('lang') or 'auto').strip().lower()
        mt = (f.mimetype or '').lower()
        suffix = '.ogg' if 'ogg' in mt else '.m4a' if ('mp4' in mt or 'aac' in mt) else '.wav' if 'wav' in mt else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp)
            path = tmp.name

        def work():
            try:
                res = self.voice.transcribe(path, lang_pref, self.chatbot)
                return {'success': True, 'transcript': res["text"], 'language': res["language"],
                        'language_name': LANG_NAMES.get(res["language"], res["language"] or ''),
                        'source': res["source"], 'uncertain': bool(res.get("uncertain"))}
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass

        return _job_start(work)


_news_prev_init = WebUI.__init__


def _news_init(self):
    _news_prev_init(self)
    if FLASK_AVAILABLE:
        _news_add_routes(self)


WebUI.__init__ = _news_init

NEWS_VOICE_HTML = """
<div class="oc-row" style="justify-content:center;margin-top:12px">
  <button type="button" class="mic-btn" id="news-mic" title="Record a voice message">🎤</button>
  <button type="button" class="mic-btn mini" id="news-speak" title="Read the result out loud">🔊</button>
  <select id="news-lang" title="Language you speak (results are shown in it too)"><option value="auto">Auto-detect</option></select>
</div>
<div class="voice-text" id="news-voice-label" style="text-align:center"></div>
"""

NEWS_VOICE_JS = r"""
(function () {
  const $ = id => document.getElementById(id);
  const ni = $('news-input'), mic = $('news-mic'), spk = $('news-speak'), sel = $('news-lang'), vl = $('news-voice-label');
  if (!ni || !mic || !sel) return;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  // ---------- searchable language picker (same list as the voice picker) ----------
  function fillLangs() {
    fetch('/api/voice_languages').then(r => r.json()).then(d => {
      sel.innerHTML = '<option value="auto">Auto-detect</option>' +
        d.languages.map(l => '<option value="' + l.code + '">' + l.name + '</option>').join('');
      let saved = 'auto'; try { saved = localStorage.getItem('pipNewsLang') || 'auto'; } catch (e) {}
      sel.value = saved;
      combo();
    }).catch(combo);
  }
  let comboDone = false;
  function combo() {
    if (comboDone) return; comboDone = true;
    if (window.__pipLangCombo) window.__pipLangCombo(sel, false);
  }
  sel.addEventListener('change', () => { try { localStorage.setItem('pipNewsLang', sel.value); } catch (e) {} });
  fillLangs();

  // ---------- remember the last result so 🔊 can read it ----------
  const base = window.fetch.bind(window);
  window.fetch = async function (url, opts) {
    const res = await base(url, opts);
    if (url === '/api/detect_fake_news') {
      try { const d = await res.clone().json(); if (d && d.success) window.__newsResult = d; } catch (e) {}
    }
    return res;
  };
  function speechText(d) {
    const f = d.fact_check || {};
    return [d.verdict, 'Credibility score ' + d.credibility_score + ' out of 100.',
            f.summary, d.recommendation].filter(Boolean).join(' ');
  }
  spk.addEventListener('click', () => {
    const d = window.__newsResult;
    if (!d) { vl.textContent = 'Run a check first, then tap 🔊 to hear it.'; return; }
    if (!window.speechSynthesis) { vl.textContent = 'This browser cannot read text aloud.'; return; }
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(speechText(d).slice(0, 1500));
    const lang = sel.value !== 'auto' ? sel.value : (d.detected_language || '');
    u.lang = lang || navigator.language || 'en-US';
    speechSynthesis.speak(u);
  });

  // ---------- shared voice recorder: mic button -> transcript ----------
  function voice(btn, status, onText) {
    let rec = null, stream = null, chunks = [], actx = null, timer = null, busy = false;
    const setSt = t => { if (status) status.textContent = t; };
    function cleanup() {
      clearInterval(timer); timer = null;
      if (stream) stream.getTracks().forEach(t => t.stop()); stream = null;
      if (actx) { try { actx.close(); } catch (e) {} actx = null; }
      btn.classList.remove('rec');
      btn.textContent = btn.dataset.icon || '🎤';
    }
    async function start() {
      if (busy) return;
      if (!navigator.mediaDevices || !window.MediaRecorder) { setSt('Recording is not supported in this browser.'); return; }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: window.__pipAudioConstraints ? window.__pipAudioConstraints() : { channelCount: 1 }
        });
      } catch (e) { setSt('Please allow microphone access.'); return; }
      chunks = [];
      const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'].find(t => MediaRecorder.isTypeSupported(t)) || '';
      const opts = { audioBitsPerSecond: 64000 }; if (mime) opts.mimeType = mime;
      rec = new MediaRecorder(stream, opts);
      rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
      rec.onstop = async () => {
        const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
        const n = blob.size;
        cleanup();
        if (n > 800) send(blob);
      };
      rec.start();
      btn.classList.add('rec'); btn.textContent = '⏹';
      const t0 = Date.now();
      timer = setInterval(() => {
        const el = Date.now() - t0;
        if (el > 120000) { try { rec.stop(); } catch (e) {} return; }
        setSt('🔴 Recording… ' + Math.round(el / 1000) + 's — tap ⏹ to send, Esc cancels');
      }, 400);
    }
    function stop(cancel) {
      if (!rec || rec.state !== 'recording') { cleanup(); return; }
      if (cancel) { try { rec.onstop = null; } catch (e) {} }
      try { rec.stop(); } catch (e) { cleanup(); }
      if (cancel) { chunks = []; cleanup(); }
    }
    async function send(blob) {
      busy = true;
      setSt('Transcribing…');
      try {
        const fd = new FormData();
        const t = blob.type || '';
        fd.append('audio', blob, 'voice.' + (t.includes('mp4') ? 'm4a' : t.includes('ogg') ? 'ogg' : 'webm'));
        const vl2 = $('voice-lang');
        fd.append('lang', sel.value !== 'auto' ? sel.value : (vl2 ? vl2.value : 'auto'));
        const r = await fetch('/api/stt_start', { method: 'POST', body: fd });
        const s = JSON.parse(await r.text());
        if (!s.job) throw new Error(s.error || 'Could not start.');
        let fails = 0, res = null;
        while (!res) {
          await sleep(900);
          try {
            const p = JSON.parse(await (await fetch('/api/job/' + s.job)).text());
            fails = 0;
            if (p.done) res = p.result;
          } catch (e) { if (++fails > 90) throw new Error('Lost connection to the server.'); }
        }
        if (!res.success) throw new Error(res.error || 'Transcription failed.');
        onText(res);
      } catch (e) {
        setSt('Error: ' + e.message);
      } finally {
        busy = false;
      }
    }
    btn.dataset.icon = btn.textContent;
    btn.addEventListener('click', () => {
      if (rec && rec.state === 'recording') stop(false); else start();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && rec && rec.state === 'recording') stop(true);
    });
  }

  // ---------- Fake News mic ----------
  voice(mic, vl, res => {
    ni.value = res.transcript;
    vl.innerHTML = '🎙️ Heard: “' + esc(res.transcript) + '”' +
      (res.language_name ? ' · ' + esc(res.language_name) : '') + (res.uncertain ? ' · not sure' : '');
    if (window.detectFakeNews) window.detectFakeNews();
  });

  // ---------- same mic for the image-analysis agent ----------
  const iaSend = $('ia-send'), iaInp = $('ia-input');
  const iaComp = iaSend && iaSend.closest('.composer');
  if (iaComp && iaInp && !iaComp.querySelector('.ia-mic')) {
    const mb = document.createElement('button');
    mb.type = 'button'; mb.className = 'mic-btn mini ia-mic'; mb.title = 'Record a voice message'; mb.textContent = '🎤';
    iaComp.insertBefore(mb, iaSend);
    const iaStatus = document.createElement('div');
    iaStatus.className = 'voice-text'; iaStatus.style.cssText = 'text-align:center;flex-basis:100%';
    iaComp.parentNode.appendChild(iaStatus);
    voice(mb, iaStatus, res => {
      iaInp.value = res.transcript;
      iaInp.dispatchEvent(new Event('input', { bubbles: true }));
      iaStatus.textContent = '🎙️ Heard: “' + res.transcript + '”' +
        (res.language_name ? ' · ' + res.language_name : '') + (res.uncertain ? ' · not sure' : '');
      if (iaSend) iaSend.click();
    });
  }
})();
"""

_news_prev_get_html = WebUI.get_html


def _news_get_html(self) -> str:
    html = _news_prev_get_html(self)
    html = html.replace('<div class="output" id="news-output"></div>',
                        NEWS_VOICE_HTML + '<div class="output" id="news-output"></div>', 1)
    return html.replace('</body>', '<script>' + NEWS_VOICE_JS + '</script></body>', 1)


WebUI.get_html = _news_get_html

# ============================================================================
# GLOBAL SPEECH CONTROL - pause / resume / stop for every read-aloud
# A small floating pill appears whenever the AI is speaking (chat, fake news,
# image chat, math, voice replies - anything using speechSynthesis).
# ============================================================================
SPEECH_CTL_JS = r"""
(function () {
  if (window.__pipSpeechCtl) return;
  window.__pipSpeechCtl = true;
  let ctl = null;
  function build() {
    if (ctl) return ctl;
    ctl = document.createElement('div');
    ctl.id = 'pip-speech-ctl';
    ctl.style.cssText = 'position:fixed;bottom:18px;right:18px;z-index:99999;gap:8px;align-items:center;' +
      'background:rgba(24,18,38,.95);border:1px solid rgba(221,132,72,.7);border-radius:999px;' +
      'padding:8px 14px;box-shadow:0 6px 24px rgba(0,0,0,.5);color:#eee;font-size:.9em;display:none';
    ctl.innerHTML = '<span>🔊 Speaking</span>' +
      '<button type="button" id="psc-pause">⏸ Pause</button>' +
      '<button type="button" id="psc-stop">⏹ Stop</button>';
    ctl.querySelectorAll('button').forEach(b => {
      b.style.cssText = 'padding:5px 12px;border-radius:999px;cursor:pointer;' +
        'border:1px solid rgba(221,132,72,.6);background:rgba(0,0,0,.5);color:#ffaa5c;font-size:.9em';
    });
    const pauseBtn = ctl.querySelector('#psc-pause');
    const stopBtn = ctl.querySelector('#psc-stop');
    pauseBtn.onclick = () => {
      if (speechSynthesis.paused) { speechSynthesis.resume(); pauseBtn.textContent = '⏸ Pause'; }
      else { speechSynthesis.pause(); pauseBtn.textContent = '▶ Resume'; }
    };
    stopBtn.onclick = () => { speechSynthesis.cancel(); };
    document.body.appendChild(ctl);
    return ctl;
  }
  setInterval(() => {
    const ss = window.speechSynthesis;
    if (!ss) return;
    const on = ss.speaking || ss.pending || ss.paused;
    const c = build();
    c.style.display = on ? 'flex' : 'none';
    if (!on) { const p = c.querySelector('#psc-pause'); if (p) p.textContent = '⏸ Pause'; }
  }, 400);
})();
"""

_speech_prev_get_html = WebUI.get_html


def _speech_get_html(self) -> str:
    html = _speech_prev_get_html(self)
    return html.replace('</body>', '<script>' + SPEECH_CTL_JS + '</script></body>', 1)


WebUI.get_html = _speech_get_html

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    import subprocess, time, urllib.request

    print("\n🔧 Initializing system...")
    try:
        from google.colab import userdata
        os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
        print("✓ Gemini key loaded from Colab Secrets")
    except Exception as e:
        print("⚠️ Could not read GEMINI_API_KEY:", e)
    ui = WebUI()
    print("✅ System initialized\n")

    # 1) Start the website in the background
    threading.Thread(
        target=lambda: ui.app.run(host="0.0.0.0", port=ui.port, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    time.sleep(2)

    # 2) Download cloudflared (free, no account, no token)
    if not os.path.exists("cloudflared"):
        print("⬇️  Downloading cloudflared...")
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "cloudflared",
        )
        os.chmod("cloudflared", 0o755)

    # 3) Open the tunnel and wait until it is really connected
    proc = subprocess.Popen(
        ["./cloudflared", "tunnel", "--url", f"http://localhost:{ui.port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    public_url = None
    connected = False
    start = time.time()
    for line in proc.stdout:
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
        if m:
            public_url = m.group(0)
        if "Registered tunnel connection" in line:
            connected = True
        if public_url and connected:
            break
        if time.time() - start > 90:
            break
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()
    if public_url and connected:
        time.sleep(15)  # give the new address time to become reachable

    # 4) Backup link made by Colab itself (only works in your own Google account)
    colab_url = None
    try:
        from google.colab.output import eval_js
        colab_url = eval_js(f"google.colab.kernel.proxyPort({ui.port})")
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("YOUR WEBSITE IS READY:")
    if public_url:
        print(f"🌐 Public link (anyone): {public_url}")
        if not connected:
            print("   ⚠️ The tunnel did not confirm it connected. If this link fails, use the backup.")
    if colab_url:
        print(f"🔒 Backup link (you only, same Google account): {colab_url}")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        proc.terminate()