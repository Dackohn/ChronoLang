package chrono.lang.chronolangserver.controllers;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.Map;

@RestController
public class ParsingController {
    @PostMapping("/parse")
    public ResponseEntity<String> parseChronoCode(@RequestBody Map<String, String> request) {
        String code = request.get("code");
        // Logging
        System.out.println(code);
        try {
            ProcessBuilder pb = new ProcessBuilder("python3", "cl/Interpreter/test.py", code);
            pb.redirectErrorStream(true);

            Process process = pb.start();
            String result = new String(process.getInputStream().readAllBytes());

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return ResponseEntity.ok(result);
            } else {
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body("Error parsing code: " + result);
            }
        } catch (IOException | InterruptedException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Execution failed: " + e.getMessage());
        }
    }
}
