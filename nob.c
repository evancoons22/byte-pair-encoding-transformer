#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_DIR "./build/"
#define SRC_DIR "./src/"


int main(int argc, char **argv)
{

    NOB_GO_REBUILD_URSELF(argc, argv);

    Nob_Cmd cmd = {0};

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [bpe|transformer|markov|markovforward]\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "bpe") == 0) { 
        nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-o", BUILD_DIR"bpe", SRC_DIR"bpe.c");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/bpe");
        if (!nob_cmd_run_sync_and_reset(&cmd)) ;
    } else if (strcmp(argv[1], "transformer") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"transformer", SRC_DIR"transformer.c", "-lm");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/transformer");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else if (strcmp(argv[1], "markov") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"markov", SRC_DIR"markov.c", "-lm");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/markov");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else if (strcmp(argv[1], "markovforward") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"markovforward", SRC_DIR"markov_forward.c");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/markovforward");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else {  
        fprintf(stderr, "Usage: %s [bpe|transformer|markov|markovforward]\n", argv[0]);
        return 1;
    } 
    return 0;
}
