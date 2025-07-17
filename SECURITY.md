# Security Policy

## Reporting security issues:

Please report security issues privately using [the vulnerability submission form](https://github.com/vllm-project/vllm/security/advisories/new).

## Issue triage

Reports will then be triaged by the [vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html).

## Threat model

Please see the [Security Guide in the vLLM documentation](https://docs.vllm.ai/en/latest/usage/security.html) for more information on vLLM's security assumptions and recommendations.

Please see [PyTorch's Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md) for more information and recommendations on how to securely interact with models.

## Issue severity

We will determine the risk of each issue, taking into account our experience dealing with past issues, versions affected, common defaults, and use cases. We use the following severity categories:

### CRITICAL Severity:
Vulnerabilities that allow remote attackers to execute arbitrary code, take full control of the system, or significantly compromise confidentiality, integrity, or availability without any interaction or privileges needed, examples include remote code execution via network, deserialization issues that allow exploit chains. Generally those issues which are rated as CVSS  ≥ 9.0.

### HIGH Severity:
Serious security flaws that allow elevated impact—like RCE in specific, limited contexts or significant data loss—but require advanced conditions or some trust, examples include RCE in advanced deployment modes (e.g. multi-node), or high impact issues where some sort of privileged network access is required. These issues typically have CVSS scores between 7.0 and 8.9

### MODERATE Severity:
Vulnerabilities that cause denial of service or partial disruption, but do not allow arbitrary code execution or data breach and have limited impact. These issues have a CVSS rating between 4.0 and 6.9

### LOW Severity:
Minor issues such as informational disclosures, logging errors, non-exploitable flaws, or weaknesses that require local or high-privilege access and offer negligible impact. Examples include side channel attacks or hash collisions. These issues often have CVSS scores less than 4.0

## Prenotification policy

For certain security issues of CRITICAL, HIGH, or MODERATE severity level, we may prenotify certain organizations or vendors, which ship vLLM.

* This prenotification is often in the form of adding their security contacts to the github security advisory, typically few days before release.

* If you wish to be added to the prenotification group, please send an email copying all the members of the [vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html). Each vendor contact will be analyzed on a case to case basis.

* We may withdraw notifying certain organisations from future prenotifications if they leak issues before they are public or over time do not add value.
