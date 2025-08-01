<?php
// Nhận headers gửi đến
$headers = getallheaders();

// Lấy token từ header 'X-Access-Token'
$received_token = $headers['X-Access-Token'] ?? '';

// Token đúng mà bạn định sẵn
$expected_token = 'abc123XYZ'; // 👉 Bạn có thể thay bằng chuỗi khác, càng khó đoán càng tốt

// Kiểm tra token
if ($received_token !== $expected_token) {
    http_response_code(403);
    echo 'Forbidden';
    exit;
}

$file_path = 'drive-folder.json';

if (!file_exists($file_path)) {
    http_response_code(404);
    echo 'File not found';
    exit;
}

header('Content-Type: application/json');
readfile($file_path);
?>
