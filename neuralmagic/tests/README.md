**neuralmagic/tests**

This directory contains a set of `*.txt` files that are used by the build and
test system to skip certain tests for certain build targets in order to 
optimize the amount of test execution time required for different scenarios.

Overall, there are four test 'cycles'/triggers in the build system:

* remote-push — occurs on any branch & commit being pushed to the repo
* release — currently only triggered manually
* nightly/weekly — basically the same job, runs on a schedule (weekly runs on Sun, nightly runs other days)

There is a list of test cases associated with each trigger, and a broadly encompassing one:
* skip-almost-all.txt (this is used for rapid GHA dev work to run fast)
* skip-for-remote-push.txt
* skip-for-release.txt
* skip-for-nightly.txt
* skip-for-weekly.txt

Particularly long-running or less critical tests should not be run during
a remote push, but should probably be run against nightly/weekly builds
and a final release build.  In such a scenario, to get your test to run for the 
release and nightly/weekly triggers, and skip it for other triggers, add your 
test (file) to the following skip lists:
* skip-almost-all.txt
* skip-for-remote-push.txt

This will basically mean your test is only skipped during remote-push, 
and will run for all other triggers.
