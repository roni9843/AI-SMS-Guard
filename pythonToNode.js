const { spawn } = require("child_process");

function classifySMS(smsText) {
    return new Promise((resolve, reject) => {
        const process = spawn("python", ["sction.py"], { env: { PYTHONIOENCODING: "utf-8" } });


        process.stdout.on("data", (data) => {
            resolve(data.toString().trim());
        });

        process.stderr.on("data", (data) => {
            console.error("Python Error:", data.toString());
            reject(data.toString());
        });

        process.on("error", (error) => {
            console.error("Failed to start process:", error);
            reject(error);
        });
    });
}

// Example Usage
const sms = "Your OTP for the transaction is 123456";
classifySMS(sms)
    .then((result) => console.log("Prediction:", result))
    .catch((error) => console.error("Error:", error));
