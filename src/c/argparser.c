
#include <stdio.h>
#include <string.h>

#define MAX_STR 50

typedef struct ParsedArguments {
    char rtsp_address[MAX_STR];
    char codec[MAX_STR];
    unsigned int debug_level;
    int display;
} ParsedArgs;

ParsedArgs DefaultParsedArgs = {"rtsp://127.0.0.1:8554/stream",
                                "h264",
                                0,
                                1
};

int parse_arguments(ParsedArgs *p, char *argv[]) {

    strcpy(p->rtsp_address, argv[1]);
    strcpy(p->codec, argv[2]);
    p->debug_level = (unsigned int) *argv[3];
    p->display = (strcmp(argv[4], "-d") == 0) ? 1 : 0;
    return 0;
}

int display_arguments(ParsedArgs *p) {
    printf("Parsed arguments:\n\trtsp_address: %s", p->rtsp_address);
    printf("\n\tcodec: %s", p->codec);
    printf("\n\tdebug_level: %d", p->debug_level);
    printf("\n\tdisplay: %d\n", p->display);
    return 0;
}
