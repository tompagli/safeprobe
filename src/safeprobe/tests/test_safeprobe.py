"""Tests for SafeProbe package."""

import pytest
import os
import sys

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestConfig:
    def test_config_creation(self):
        from safeprobe.config import SafeProbeConfig
        config = SafeProbeConfig()
        assert config.target_model == "gpt-4.1-2025-04-14"
        assert config.iterations == 3

    def test_config_validation_no_keys(self):
        from safeprobe.config import SafeProbeConfig
        config = SafeProbeConfig(
            openai_api_key=None,
            anthropic_api_key=None,
            google_api_key=None,
            xai_api_key=None,
            azure_api_key=None,
        )
        assert config.validate() == False

    def test_config_validation_with_key(self):
        from safeprobe.config import SafeProbeConfig
        config = SafeProbeConfig(openai_api_key="test-key")
        assert config.validate() == True


class TestImports:
    def test_import_main_package(self):
        import safeprobe
        assert safeprobe.__version__ == "1.0.0"

    def test_import_attacks(self):
        from safeprobe.attacks.promptmap.attack import PromptMapAttack
        from safeprobe.attacks.pair.attack import PAIRAttack
        from safeprobe.attacks.cipherchat.attack import CipherChatAttack
        assert PromptMapAttack is not None
        assert PAIRAttack is not None
        assert CipherChatAttack is not None

    def test_import_analysis(self):
        from safeprobe.analysis.judge import JailbreakJudge
        from safeprobe.analysis.consolidator import ResultsConsolidator
        from safeprobe.analysis.report_gen import ReportGenerator
        assert JailbreakJudge is not None
        assert ResultsConsolidator is not None
        assert ReportGenerator is not None

    def test_import_utils(self):
        from safeprobe.utils.logging import get_logger
        from safeprobe.utils.unified_lib import ensure_json_file, append_json_rows
        assert get_logger is not None


class TestCiphers:
    def test_caesar_encode_decode(self):
        from safeprobe.attacks.cipherchat.attack import ENCODE_EXPERTS
        caesar = ENCODE_EXPERTS["caesar"]
        original = "hello world"
        encoded = caesar.encode(original)
        decoded = caesar.decode(encoded)
        assert decoded == original
        assert encoded != original

    def test_atbash_encode_decode(self):
        from safeprobe.attacks.cipherchat.attack import ENCODE_EXPERTS
        atbash = ENCODE_EXPERTS["atbash"]
        original = "hello"
        encoded = atbash.encode(original)
        decoded = atbash.decode(encoded)
        assert decoded == original

    def test_unchanged(self):
        from safeprobe.attacks.cipherchat.attack import ENCODE_EXPERTS
        unch = ENCODE_EXPERTS["unchange"]
        assert unch.encode("test") == "test"
        assert unch.decode("test") == "test"


class TestAttackInit:
    def test_promptmap_init(self):
        from safeprobe.attacks.promptmap.attack import PromptMapAttack
        attack = PromptMapAttack()
        assert attack.name == "PromptMap"
        assert attack.category == "manual"

    def test_pair_init(self):
        from safeprobe.attacks.pair.attack import PAIRAttack
        attack = PAIRAttack()
        assert attack.name == "PAIR"
        assert attack.category == "automated"

    def test_cipherchat_init(self):
        from safeprobe.attacks.cipherchat.attack import CipherChatAttack
        attack = CipherChatAttack()
        assert attack.name == "CipherChat"


class TestConsolidator:
    def test_asr_computation(self):
        import pandas as pd
        from safeprobe.analysis.consolidator import ResultsConsolidator
        df = pd.DataFrame({
            "attack_tool": ["PAIR", "PAIR", "PAIR", "PAIR"],
            "attack_successful": [True, False, True, False],
        })
        asr = ResultsConsolidator.compute_asr(df)
        assert asr == 50.0

    def test_robustness_score(self):
        import pandas as pd
        from safeprobe.analysis.consolidator import ResultsConsolidator
        df = pd.DataFrame({
            "attack_tool": ["PromptMap", "PAIR", "CipherChat"],
            "attack_successful": [False, False, True],
        })
        score = ResultsConsolidator.compute_robustness_score(df)
        # PromptMap (1pt resisted) + PAIR (3pt resisted) + CipherChat (0pt) = 4/9 * 100
        assert abs(score - (4 / 9 * 100)) < 0.1


class TestUnifiedLib:
    def test_ensure_json_file(self, tmp_path):
        from safeprobe.utils.unified_lib import ensure_json_file
        import json
        path = str(tmp_path / "test.json")
        ensure_json_file(path)
        with open(path) as f:
            data = json.load(f)
        assert data == []

    def test_append_json_rows(self, tmp_path):
        from safeprobe.utils.unified_lib import append_json_rows
        import json
        path = str(tmp_path / "test.json")
        append_json_rows(path, [{"a": 1}])
        append_json_rows(path, [{"b": 2}])
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
