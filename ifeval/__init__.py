"""Vendored IFEval verifiers.

Copied verbatim from google-research/instruction_following_eval (Apache-2.0,
see LICENSE), except that the two intra-package imports are repointed at this
package. IFEval's value is that its constraints are checked by code rather than
a judge, so the reference implementation is used as-is: reimplementing the
verifiers would silently shift the IFEval component of Y.
"""
