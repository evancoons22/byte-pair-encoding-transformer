#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_DIR "./build/"


int main(int argc, char **argv)
{

    NOB_GO_REBUILD_URSELF(argc, argv);

    Nob_Cmd cmd = {0};

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [bpe|train|markov|markovforward]\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "bpe") == 0) { 
        nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-o", BUILD_DIR"bpe", "bpe.c");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/bpe");
        if (!nob_cmd_run_sync_and_reset(&cmd)) ;
    } else if (strcmp(argv[1], "train") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"train", "train.c", "-lm");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/train");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else if (strcmp(argv[1], "markov") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"markov", "markov.c", "-lm");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/markov");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else if (strcmp(argv[1], "markovforward") == 0) {  
        nob_cmd_append(&cmd, "cc", "-Wall", "-g", "-Wextra", "-o", BUILD_DIR"markovforward", "markov_forward.c");
        (!nob_cmd_run_sync_and_reset(&cmd));
        nob_cmd_append(&cmd, "./build/markovforward");
        if (!nob_cmd_run_sync_and_reset(&cmd));
    } else {  
        fprintf(stderr, "Usage: %s [bpe|train|markov|markovforward]\n", argv[0]);
        return 1;
    } 
    return 0;
}
