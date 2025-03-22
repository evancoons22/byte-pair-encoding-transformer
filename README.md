## byte pair encoding in c
- This uses **[byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)** algorithm to compress text. 

### usage
```
$ cc -o nob nob.c // only once
$ ./nob 
```
note: `run_version_1()` is a naive approach. `run_version_2()` uses max heap and hash map to improve time

### examples
- data comes from the [gutenberg project](https://www.gutenberg.org/)
