#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
// Copyright (C) 2024-2025 Guyutongxue
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// The block to remove (Escaped for C)
const char *BLOCK = "// Copyright (C) 2024-2025 Guyutongxue\r\n"
                    "// \r\n"
                    "// This program is free software: you can redistribute it and/or modify\r\n"
                    "// it under the terms of the GNU Affero General Public License as\r\n"
                    "// published by the Free Software Foundation, either version 3 of the\r\n"
                    "// License, or (at your option) any later version.\r\n"
                    "// \r\n"
                    "// This program is distributed in the hope that it will be useful,\r\n"
                    "// but WITHOUT ANY WARRANTY; without even the implied warranty of\r\n"
                    "// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\r\n"
                    "// GNU Affero General Public License for more details.\r\n"
                    "// \r\n"
                    "// You should have received a copy of the GNU Affero General Public License\r\n"
                    "// along with this program.  If not, see <http://www.gnu.org/licenses/>.\r\n";

// Check if file has the correct extension
int is_target_file(const char *name) {
    const char *ext = strrchr(name, '.');
    if (!ext) return 0;
    const char *targets[] = {".txt", ".md", ".cpp", ".h", ".tsx", ".js", ".ts", ".css"};
    for (int i = 0; i < 8; i++) {
        if (strcmp(ext, targets[i]) == 0) return 1;
    }
    return 0;
}

void process_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    // Search for the block
    char *pos = strstr(buffer, BLOCK);
    if (pos) {
        printf("Processing: %s\n", path);
        size_t block_len = strlen(BLOCK);
        
        // Write the file back, skipping the block
        f = fopen(path, "wb");
        if (f) {
            // Write everything before the block
            fwrite(buffer, 1, pos - buffer, f);
            // Write everything after the block
            fwrite(pos + block_len, 1, (buffer + size) - (pos + block_len), f);
            fclose(f);
        }
    } else {
        int has_empty_first_line = (size >= 2 && buffer[0] == '\r' && buffer[1] == '\n');
        
        if (has_empty_first_line) {
            printf("Processing: %s (removing empty first line)\n", path);
            f = fopen(path, "wb");
            if (f) {
                // Skip the first 2 characters (\r\n) and write the rest
                fwrite(buffer + 2, 1, size - 2, f);
                fclose(f);
            }
        }
    }

    free(buffer);
}

void walk_directory(const char *base_path) {
    struct dirent *dp;
    DIR *dir = opendir(base_path);
    if (!dir) return;

    while ((dp = readdir(dir)) != NULL) {
        // 1. 忽略隐藏文件夹和 node_modules (性能关键！)
        if (dp->d_name[0] == '.' || strcmp(dp->d_name, "node_modules") == 0) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", base_path, dp->d_name);

        struct stat statbuf;
        if (stat(path, &statbuf) != 0) continue;

        if (S_ISDIR(statbuf.st_mode)) {
            walk_directory(path); // 递归
        } else if (is_target_file(path)) {
            // 2. 打印当前正在扫描的文件，确定程序没死
            printf("Scanning: %s\r", path); 
            fflush(stdout); 
            process_file(path);
        }
    }
    closedir(dir);
}


int main() {
    walk_directory(".");
    printf("Done.\n");
    return 0;
}

