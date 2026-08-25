from eval.bias_metrics import delta_beyond_noise, noise_floor


def test_noise_floor_shape():
    floor = noise_floor([3.0, 3.1, 2.9])
    assert abs(floor["mean"] - 3.0) < 1e-9
    assert floor["stdev"] > 0


def test_noise_floor_single_sample_has_zero_stdev():
    floor = noise_floor([3.0])
    assert floor["stdev"] == 0.0


def test_delta_beyond_noise_small_delta_is_noise():
    floor = {"mean": 3.0, "stdev": 0.3}
    result = delta_beyond_noise([3.0, 3.1, 2.9], [3.05, 3.0, 2.95], floor)
    assert result["beyond_noise"] is False


def test_delta_beyond_noise_large_delta_is_real():
    floor = {"mean": 3.0, "stdev": 0.05}
    result = delta_beyond_noise([4.0, 4.1, 3.9], [2.0, 2.1, 1.9], floor)
    assert result["beyond_noise"] is True
    assert result["delta"] > 0


def test_delta_beyond_noise_zero_stdev_still_has_nonzero_threshold():
    # A single-sample or degenerate-zero-variance floor shouldn't make the
    # threshold zero (any nonzero delta would then trivially "beyond noise").
    floor = {"mean": 3.0, "stdev": 0.0}
    result = delta_beyond_noise([3.0], [3.01], floor)
    assert result["beyond_noise"] is False
    assert result["noise_threshold"] >= 0.10  # rubric quantum floor, not an arbitrary 0.05


def test_delta_beyond_noise_float_noise_does_not_flip_identical_deltas():
    # Caught on a live run (eval/bias_results.json): three mathematically
    # identical -0.100 deltas landed on both sides of the >= 0.10 threshold
    # purely from float subtraction noise. The exact real values matter —
    # a fixture that happens to round the same way on both sides of the
    # boundary would pass even without the fix (verified: an earlier
    # version of this test used inputs that were tautologically safe).
    # 4.0 - 4.1 == -0.09999999999999964 (< 0.10 in raw float math, False
    # pre-fix); 3.9 - 4.0 == -0.10000000000000009 (> 0.10, True pre-fix) —
    # same -0.1 mathematically, opposite verdicts without rounding.
    floor = {"mean": 3.0, "stdev": 0.0}
    results = [
        delta_beyond_noise([4.0], [4.1], floor),
        delta_beyond_noise([4.0], [4.1], floor),
        delta_beyond_noise([3.9], [4.0], floor),
    ]
    verdicts = {r["beyond_noise"] for r in results}
    assert len(verdicts) == 1, f"identical -0.1 deltas disagreed: {results}"


def test_delta_beyond_noise_below_rubric_quantum_is_not_beyond_noise():
    # A delta smaller than the 0.10 rubric-quantum floor (education_fit's
    # weight, the smallest criterion) shouldn't read as a "real" finding.
    # Clearly below the 0.10 threshold, not sitting exactly on it, since
    # `3.0 - 3.10` lands on the wrong side of 0.10 by float noise either way.
    floor = {"mean": 3.0, "stdev": 0.0}
    result = delta_beyond_noise([3.0], [3.05], floor)
    assert result["beyond_noise"] is False
