![conch](https://user-images.githubusercontent.com/12631702/210295964-785cc63d-d697-420c-99ff-f492eb81dec9.svg)

# `conc`: better structured concurrency for go

[![Go Reference](https://pkg.go.dev/badge/github.com/sourcegraph/conc.svg)](https://pkg.go.dev/github.com/sourcegraph/conc)
[![Sourcegraph](https://img.shields.io/badge/view%20on-sourcegraph-A112FE?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAEZklEQVRoQ+2aXWgUZxSG3292sxtNN43BhBakFPyhxSujRSxiU1pr7SaGXqgUxOIEW0IFkeYighYUxAuLUlq0lrq2iCDpjWtmFVtoG6QVNOCFVShVLyxIk0DVjZLMxt3xTGTccd2ZOd/8JBHci0CY9zvnPPN+/7sCIXwKavOwAcy2QgngQiIztDSE0OwQlDPYR1ebiaH6J5kZChyfW12gRG4QVgGTBfMchMbFP9Sn5nlZL2D0JjLD6710lc+z0NfqSGTXQRQ4bX07Mq423yoBL3OSyHSvUxirMuaEvgbJWrdcvkHMoJwxYuq4INUhyuWvQa1jvdMGxAvCxJlyEC9XOBCWL04wwRzpbDoDQ7wfZJzIQLi5Eggk6DiRhZgWIAbE3NrM4A3LPT8Q7UgqAqLqTmLSHLGPkyzG/qXEczhd0q6RH+zaSBfaUoc4iQx19pIClIscrTkNZzG6gd7qMY6eC2Hqyo705ZfTf+eqJmhMzcSbYtQpOXc92ZsZjLVAL4YNUQbJ5Ttg4CQrQdGYj44Xr9m1XJCzmZusFDJOWNpHjmh5x624a2ZFtOKDVL+uNo2TuXE3bZQQZUf8gtgqP31uI94Z/rMqix+IGiRfWw3xN9dCgVx+L3WrHm4Dju6PXz/EkjuXJ6R+IGgyOE1TbZqTq9y1eo0EZo7oMo1ktPu3xjHvuiLT5AFNszUyDULtWpzE2/fEsey8O5TbWuGWwxrs5rS7nFNMWJrNh2No74s9Ec4vRNmRRzPXMP19fBMSVsGcOJ98G8N3Wl2gXcbTjbX7vUBxLaeASDQCm5Cu/0E2tvtb0Ea+BowtskFD0wvlc6Rf2M+Jx7dTu7ubFr2dnKDRaMQe2v/tcIrNB7FH0O50AcrBaApmRDVwFO31ql3pD8QW4dP0feNwl/Q+kFEtRyIGyaWXnpy1OO0qNJWHo1y6iCmAGkBb/Ru+HenDWIF2mo4r8G+tRRzoniSn2uqFLxANhe9LKHVyTbz6egk9+x5w5fK6ulSNNMhZ/Feno+GebLZV6isTTa6k5qNl5RnZ5u56Ib6SBvFzaWBBVFZzvnERWlt/Cg4l27XChLCqFyLekjhy6xJyoytgjPf7opIB8QPx7sYFiMXHPGt76m741MhCKMZfng0nBOIjmoJPsLqWHwgFpe6V6qtfcopxveR2Oy+J0ntIN/zCWkf8QNAJ7y6d8Bq4lxLc2/qJl5K7t432XwcqX5CrI34gzATWuYILQtdQPyePDK3iuOekCR3Efjhig1B1Uq5UoXEEoZX7d1q535J5S9VOeFyYyEBku5XTMXXKQTToX5Rg7OI44nbW5oKYeYK4EniMeF0YFNSmb+grhc84LyRCEP1/OurOcipCQbKxDeK2V5FcVyIDMQvsgz5gwFhcWWwKyRlvQ3gv29RwWoDYAbIofNyBxI9eDlQ+n3YgsgCWnr4MStGXQXmv9pF2La/k3OccV54JEBM4yp9EsXa/3LfO0dGPcYq0Y7DfZB8nJzZw2rppHgKgVHs8L5wvRwAAAABJRU5ErkJggg==)](https://sourcegraph.com/github.com/sourcegraph/conc)
[![Go Report Card](https://goreportcard.com/badge/github.com/sourcegraph/conc)](https://goreportcard.com/report/github.com/sourcegraph/conc)
[![codecov](https://codecov.io/gh/sourcegraph/conc/branch/main/graph/badge.svg?token=MQZTEA1QWT)](https://codecov.io/gh/sourcegraph/conc)
[![Discord](https://img.shields.io/badge/discord-chat-%235765F2)](https://discord.gg/bvXQXmtRjN)

`conc` is your toolbelt for structured concurrency in go, making common tasks
easier and safer.

```sh
go get github.com/sourcegraph/conc
```

# At a glance

- Use [`conc.WaitGroup`](https://pkg.go.dev/github.com/sourcegraph/conc#WaitGroup) if you just want a safer version of `sync.WaitGroup`
- Use [`pool.Pool`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#Pool) if you want a concurrency-limited task runner
- Use [`pool.ResultPool`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#ResultPool) if you want a concurrent task runner that collects task results
- Use [`pool.(Result)?ErrorPool`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#ErrorPool) if your tasks are fallible
- Use [`pool.(Result)?ContextPool`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#ContextPool) if your tasks should be canceled on failure
- Use [`stream.Stream`](https://pkg.go.dev/github.com/sourcegraph/conc/stream#Stream) if you want to process an ordered stream of tasks in parallel with serial callbacks
- Use [`iter.Map`](https://pkg.go.dev/github.com/sourcegraph/conc/iter#Map) if you want to concurrently map a slice
- Use [`iter.ForEach`](https://pkg.go.dev/github.com/sourcegraph/conc/iter#ForEach) if you want to concurrently iterate over a slice
- Use [`panics.Catcher`](https://pkg.go.dev/github.com/sourcegraph/conc/panics#Catcher) if you want to catch panics in your own goroutines

All pools are created with
[`pool.New()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#New)
or
[`pool.NewWithResults[T]()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#NewWithResults),
then configured with methods:

- [`p.WithMaxGoroutines()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#Pool.MaxGoroutines) configures the maximum number of goroutines in the pool
- [`p.WithErrors()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#Pool.WithErrors) configures the pool to run tasks that return errors
- [`p.WithContext(ctx)`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#Pool.WithContext) configures the pool to run tasks that should be canceled on first error
- [`p.WithFirstError()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#ErrorPool.WithFirstError) configures error pools to only keep the first returned error rather than an aggregated error
- [`p.WithCollectErrored()`](https://pkg.go.dev/github.com/sourcegraph/conc/pool#ResultContextPool.WithCollectErrored) configures result pools to collect results even when the task errored

# Goals

The main goals of the package are:
1) Make it harder to leak goroutines
2) Handle panics gracefully
3) Make concurrent code easier to read

## Goal #1: Make it harder to leak goroutines

A common pain point when working with goroutines is cleaning them up. It's
really easy to fire off a `go` statement and fail to properly wait for it to
complete.

`conc` takes the opinionated stance that all concurrency should be scoped.
That is, goroutines should have an owner and that owner should always
ensure that its owned goroutines exit properly.

In `conc`, the owner of a goroutine is always a `conc.WaitGroup`. Goroutines
are spawned in a `WaitGroup` with `(*WaitGroup).Go()`, and
`(*WaitGroup).Wait()` should always be called before the `WaitGroup` goes out
of scope.

In some cases, you might want a spawned goroutine to outlast the scope of the
caller. In that case, you could pass a `WaitGroup` into the spawning function.

```go
func main() {
    var wg conc.WaitGroup
    defer wg.Wait()

    startTheThing(&wg)
}

func startTheThing(wg *conc.WaitGroup) {
    wg.Go(func() { ... })
}
```

For some more discussion on why scoped concurrency is nice, check out [this
blog
post](https://vorpus.org/blog/notes-on-structured-concurrency-or-go-statement-considered-harmful/).

## Goal #2: Handle panics gracefully

A frequent problem with goroutines in long-running applications is handling
panics. A goroutine spawned without a panic handler will crash the whole process
on panic. This is usually undesirable.

However, if you do add a panic handler to a goroutine, what do you do with the
panic once you catch it? Some options:
1) Ignore it
2) Log it
3) Turn it into an error and return that to the goroutine spawner
4) Propagate the panic to the goroutine spawner

Ignoring panics is a bad idea since panics usually mean there is actually
something wrong and someone should fix it.

Just logging panics isn't great either because then there is no indication to the spawner
that something bad happened, and it might just continue on as normal even though your
program is in a really bad state.

Both (3) and (4) are reasonable options, but both require the goroutine to have
an owner that can actually receive the message that something went wrong. This
is generally not true with a goroutine spawned with `go`, but in the `conc`
package, all goroutines have an owner that must collect the spawned goroutine.
In the conc package, any call to `Wait()` will panic if any of the spawned goroutines
panicked. Additionally, it decorates the panic value with a stacktrace from the child
goroutine so that you don't lose information about what caused the panic.

Doing this all correctly every time you spawn something with `go` is not
trivial and it requires a lot of boilerplate that makes the important parts of
the code more difficult to read, so `conc` does this for you.

<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
type caughtPanicError struct {
    val   any
    stack []byte
}

func (e *caughtPanicError) Error() string {
    return fmt.Sprintf(
        "panic: %q\n%s",
        e.val,
        string(e.stack)
    )
}

func main() {
    done := make(chan error)
    go func() {
        defer func() {
            if v := recover(); v != nil {
                done <- &caughtPanicError{
                    val: v,
                    stack: debug.Stack()
                }
            } else {
                done <- nil
            }
        }()
        doSomethingThatMightPanic()
    }()
    err := <-done
    if err != nil {
        panic(err)
    }
}
```
</td>
<td>

```go
func main() {
    var wg conc.WaitGroup
    wg.Go(doSomethingThatMightPanic)
    // panics with a nice stacktrace
    wg.Wait()
}
```
</td>
</tr>
</table>

## Goal #3: Make concurrent code easier to read

Doing concurrency correctly is difficult. Doing it in a way that doesn't
obfuscate what the code is actually doing is more difficult. The `conc` package
attempts to make common operations easier by abstracting as much boilerplate
complexity as possible.

Want to run a set of concurrent tasks with a bounded set of goroutines? Use
`pool.New()`. Want to process an ordered stream of results concurrently, but
still maintain order? Try `stream.New()`. What about a concurrent map over
a slice? Take a peek at `iter.Map()`.

Browse some examples below for some comparisons with doing these by hand.

# Examples

Each of these examples forgoes propagating panics for simplicity. To see
what kind of complexity that would add, check out the "Goal #2" header above.

Spawn a set of goroutines and waiting for them to finish:

<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
func main() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            // crashes on panic!
            doSomething()
        }()
    }
    wg.Wait()
}
```
</td>
<td>

```go
func main() {
    var wg conc.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Go(doSomething)
    }
    wg.Wait()
}
```
</td>
</tr>
</table>

Process each element of a stream in a static pool of goroutines:

<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
func process(stream chan int) {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for elem := range stream {
                handle(elem)
            }
        }()
    }
    wg.Wait()
}
```
</td>
<td>

```go
func process(stream chan int) {
    p := pool.New().WithMaxGoroutines(10)
    for elem := range stream {
        elem := elem
        p.Go(func() {
            handle(elem)
        })
    }
    p.Wait()
}
```
</td>
</tr>
</table>

Process each element of a slice in a static pool of goroutines:

<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
func process(values []int) {
    feeder := make(chan int, 8)

    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for elem := range feeder {
                handle(elem)
            }
        }()
    }

    for _, value := range values {
        feeder <- value
    }
    close(feeder)
    wg.Wait()
}
```
</td>
<td>

```go
func process(values []int) {
    iter.ForEach(values, handle)
}
```
</td>
</tr>
</table>

Concurrently map a slice:

<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
func concMap(
    input []int,
    f func(int) int,
) []int {
    res := make([]int, len(input))
    var idx atomic.Int64

    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()

            for {
                i := int(idx.Add(1) - 1)
                if i >= len(input) {
                    return
                }

                res[i] = f(input[i])
            }
        }()
    }
    wg.Wait()
    return res
}
```
</td>
<td>

```go
func concMap(
    input []int,
    f func(*int) int,
) []int {
    return iter.Map(input, f)
}
```
</td>
</tr>
</table>

Process an ordered stream concurrently:


<table>
<tr>
<th><code>stdlib</code></th>
<th><code>conc</code></th>
</tr>
<tr>
<td>

```go
func mapStream(
    in chan int,
    out chan int,
    f func(int) int,
) {
    tasks := make(chan func())
    taskResults := make(chan chan int)

    // Worker goroutines
    var workerWg sync.WaitGroup
    for i := 0; i < 10; i++ {
        workerWg.Add(1)
        go func() {
            defer workerWg.Done()
            for task := range tasks {
                task()
            }
        }()
    }

    // Ordered reader goroutines
    var readerWg sync.WaitGroup
    readerWg.Add(1)
    go func() {
        defer readerWg.Done()
        for result := range taskResults {
            item := <-result
            out <- item
        }
    }()

    // Feed the workers with tasks
    for elem := range in {
        resultCh := make(chan int, 1)
        taskResults <- resultCh
        tasks <- func() {
            resultCh <- f(elem)
        }
    }

    // We've exhausted input.
    // Wait for everything to finish
    close(tasks)
    workerWg.Wait()
    close(taskResults)
    readerWg.Wait()
}
```
</td>
<td>

```go
func mapStream(
    in chan int,
    out chan int,
    f func(int) int,
) {
    s := stream.New().WithMaxGoroutines(10)
    for elem := range in {
        elem := elem
        s.Go(func() stream.Callback {
            res := f(elem)
            return func() { out <- res }
        })
    }
    s.Wait()
}
```
</td>
</tr>
</table>

# Status

This package is currently pre-1.0. There are likely to be minor breaking
changes before a 1.0 release as we stabilize the APIs and tweak defaults.
Please open an issue if you have questions, concerns, or requests that you'd
like addressed before the 1.0 release. Currently, a 1.0 is targeted for 
March 2023.
