## Text generation in C -- byte pair encoding, transformer, markov chain
- Use [byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) algorithm to compress text
- Randomly initialized [transformer architecture](https://arxiv.org/abs/1706.03762) for test forward pass with input
- Create a [markov chain](https://en.wikipedia.org/wiki/Markov_chain) from example text and byte pair encoding
- Generate text with token inputs and markov chain. 

### usage
```
# build with nob
$ cc -o nob nob.c
$ ./nob bpe # create the byte pair encoding
$ ./nob transformer # test output of randomly initialized transformer (don't have training yet)
$ ./nob markov # create the markov chain
$ ./nob markovforward # use the markov chain to generate text
```

### example
- using example *Crime and Punishment* by Dostoevsky, specified in bpe.c and markov.c
- text comes from the [gutenberg project](https://www.gutenberg.org/)
```
# ... above commands

$ ./nob markovforward

Enter seed text> the
ziat does thin throad, wasured
Oh, th the dried
beer! Youll tell me istairl an and of ous is he ge! crosses, inter and
Oh, spot as wer dayster proved irrible bur mush to his not flashe mome... all
```
