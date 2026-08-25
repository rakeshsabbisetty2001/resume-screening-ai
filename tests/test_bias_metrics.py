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


def test_delta_beyond_noise_below_rubric_quantum_is_not_beyond_noise():
    # A delta smaller than the 0.10 rubric-quantum floor (education_fit's
    # weight, the smallest criterion) shouldn't read as a "real" finding.
    # Clearly below the 0.10 threshold, not sitting exactly on it, since
    # `3.0 - 3.10` lands on the wrong side of 0.10 by float noise either way.
    floor = {"mean": 3.0, "stdev": 0.0}
    result = delta_beyond_noise([3.0], [3.05], floor)
    assert result["beyond_noise"] is False
