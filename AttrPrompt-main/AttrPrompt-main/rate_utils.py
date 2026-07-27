"""Canonical Metattack-rate parsing shared by loaders and run scripts."""
import argparse
from decimal import Decimal, InvalidOperation


RATE_QUANTUM = Decimal('0.01')


def canonical_rate_decimal(value):
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid perturbation rate: {value!r}") from exc

    if not decimal_value.is_finite():
        raise ValueError(f"Perturbation rate must be finite: {value!r}")
    quantized = decimal_value.quantize(RATE_QUANTUM)
    if decimal_value != quantized:
        raise ValueError(
            f"Perturbation rate supports at most two decimals: {value!r}")
    if quantized < 0 or quantized > 1:
        raise ValueError(
            f"Perturbation rate must be between 0 and 1: {value!r}")
    if quantized == 0:
        quantized = Decimal('0.00')
    return quantized


def canonical_rate(value):
    return float(canonical_rate_decimal(value))


def rate_token(value):
    """Dataset filename token, for example 0.1 -> '0.10'."""
    return format(canonical_rate_decimal(value), '.2f')


def rate_tag(value):
    """Directory-safe canonical tag, for example 0.1 -> 'M0p10'."""
    return f"M{rate_token(value).replace('.', 'p')}"


def canonical_rate_tokens(values):
    tokens = []
    seen = {}
    for value in values:
        token = rate_token(value)
        if token in seen:
            raise ValueError(
                f"Duplicate perturbation rate after normalization: "
                f"{seen[token]!r} and {value!r} both mean {token}")
        seen[token] = value
        tokens.append(token)
    if not tokens:
        raise ValueError("At least one perturbation rate is required.")
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rates', nargs='+')
    args = parser.parse_args()
    try:
        print(' '.join(canonical_rate_tokens(args.rates)))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
