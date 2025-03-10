#include <stdio.h> 
#include <string.h>
#include <stdlib.h>
#include <assert.h>

// building a byte pair encoding in c
// eventually will feed this encoding to train something

#define DA_INIT_CAP

#define da_append(da, item)                                                          \
    do {                                                                                 \
        if ((da)->count >= (da)->capacity) {                                             \
            (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;   \
            (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items)); \
            assert((da)->items != NULL && "Buy more RAM lol");                       \
        }                                                                                \
                                                                                         \
        (da)->items[(da)->count++] = (item);                                             \
    } while (0)

typedef struct { 
	char text[2];
	int count;
} Pair;

typedef struct { 
    Pair* items;
    int count;
    int capacity;
} Map;

int find_str(Map* arr, char *x) { 
    for (size_t i = 0; i < arr->count; i ++)  { 
        if (memcmp(arr->items[i].text,x,2) == 0) { 
            //printf("compared and found the right value: %d, %s",i, arr->items[i].text );
            return i;
        } 
    } 
    return -1;
} 


Map build_map(char* text_test) { 
	size_t text_len = strlen(text_test);
    Map arr[] = {0};
	printf("The size of the string is %zu\n", text_len);

    for (size_t i = 0; i < text_len - 1; i ++)  { 
        char e[] = { text_test[i], text_test[i+1] };
        Pair p = {
            .text = { text_test[i], text_test[i+1] },
            .count = 1
        };
        int v = find_str(arr, e);
        if (v < 0) { 
            da_append(arr, p);
        } else {  
            arr->items[v].count += 1;
        } 
    } 

    //print map
  //  for (size_t i = 0; i < arr->count; i ++)  { 
  //      printf("%c%c: %d\n", arr->items[i].text[0], arr->items[i].text[1], arr->items[i].count);
  //  } 
    return *arr;

} 

int main() { 

//	Pair kv = { 
//		.text = "ab",
//		.count = 1,
//	};

	char* text_test = "This is a test input";

    Map map = build_map(text_test);

    return 0;

} 
