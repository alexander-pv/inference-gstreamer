
#include "argparser.c"

// Get terminal arguments wrapped in a ParsedArguments struct
int parse_arguments(ParsedArgs *p, char *argv[]);

// Display parsed arguments
int display_arguments(ParsedArgs *p);

// ParsedArgs custom data type
typedef struct ParsedArguments ParsedArgs;
