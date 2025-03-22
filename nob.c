#define NOB_IMPLEMENTATION
#include "nob.h"

#define BUILD_DIR "./build/"

int main(int argc, char **argv)
{

    NOB_GO_REBUILD_URSELF(argc, argv);

    Nob_Cmd cmd = {0};

    #ifdef _WIN32
        nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-o", BUILD_DIR"bpe", "bpe.c");
        if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
        nob_cmd_append(&cmd, "./build/bpe.exe");
        if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
    #else
        nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-O3", "-o", BUILD_DIR"bpe", "bpe.c");
        if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
        nob_cmd_append(&cmd, "./build/bpe");
        if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
    #endif


   // for when compiling the hashmap too
   //
   // if (argc < 2) {
   //     fprintf(stderr, "Usage: %s [bpe|hash]\n", argv[0]);
   //     return 1;
   // }
   // if (strcmp(argv[1], "bpe") == 0) { 
   // } 
   // else if (strcmp(argv[1], "hash") == 0) { 
   //     nob_cmd_append(&cmd, "cc", "-Wall", "-Wextra", "-o", BUILD_DIR"hash", "hashmap.c");
   //     if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
   //     nob_cmd_append(&cmd, "./build/hash");
   //     if (!nob_cmd_run_sync_and_reset(&cmd)) return 1;
   // } 


    return 0;
}
