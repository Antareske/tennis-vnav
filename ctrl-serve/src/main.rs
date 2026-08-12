//! ctrl-serve — SG2002 网球导航控制服务
//!
//! 手机连小车热点后访问 http://192.168.4.1 即可控制数采。
//! 与导航进程通过 /tmp/vnav/ 目录下的文件通信（零依赖、零协议库）。
//!
//! 端点:
//!   GET  /                   控制页面（HTML 编译进二进制）
//!   GET  /status             读取 status.json 原样返回
//!   POST /cmd/{start,abort,clear}   写入命令文件

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

const BIND_ADDR: &str = "0.0.0.0:80";
const VNAV_DIR: &str = "/tmp/vnav";
const STATUS_FILE: &str = "/tmp/vnav/status.json";
const CMD_FILE: &str = "/tmp/vnav/cmd.txt";

static INDEX_HTML: &str = include_str!("../ui/index.html");

fn main() {
    // 启动目录（导航进程与命令文件共用）
    if let Err(e) = fs::create_dir_all(VNAV_DIR) {
        eprintln!("[ctrl-serve] mkdir {}: {}", VNAV_DIR, e);
    }

    // 绑定端口，带重试（开机时 AP 网络可能未就绪）
    let listener = bind_with_retry();
    eprintln!("[ctrl-serve] listening on http://{}", BIND_ADDR);

    // 单线程顺序处理（手机单用户场景足够）
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
                handle_conn(stream);
            }
            Err(e) => eprintln!("[ctrl-serve] accept: {}", e),
        }
    }
}

fn bind_with_retry() -> TcpListener {
    for i in 0..30 {
        match TcpListener::bind(BIND_ADDR) {
            Ok(l) => return l,
            Err(e) => {
                eprintln!("[ctrl-serve] bind {} 失败 (第 {} 次): {}", BIND_ADDR, i + 1, e);
                thread::sleep(Duration::from_secs(2));
            }
        }
    }
    panic!("无法绑定 {} 端口", BIND_ADDR);
}

fn handle_conn(mut stream: TcpStream) {
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    if n == 0 {
        return;
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let first_line = req.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        respond(&mut stream, 400, "application/json", b"{\"error\":\"bad request\"}");
        return;
    }
    let (method, path) = (parts[0], parts[1]);

    match (method, path) {
        ("GET", "/") => {
            respond(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML.as_bytes());
        }
        ("GET", "/status") => match fs::read(STATUS_FILE) {
            Ok(data) => respond(&mut stream, 200, "application/json; charset=utf-8", &data),
            Err(_) => respond(
                &mut stream,
                200,
                "application/json; charset=utf-8",
                "{\"phase\":\"NO_NAV\",\"error\":\"导航进程未运行\"}".as_bytes(),
            ),
        },
        ("POST", path) if path.starts_with("/cmd/") => {
            let cmd = &path[5..];
            match cmd {
                "start" | "abort" | "clear" => {
                    match write_cmd(cmd) {
                        Ok(()) => respond(&mut stream, 200, "application/json; charset=utf-8", b"{\"ok\":true}"),
                        Err(e) => {
                            let msg = format!("{{\"ok\":false,\"error\":\"{}\"}}", e);
                            respond(&mut stream, 500, "application/json; charset=utf-8", msg.as_bytes());
                        }
                    }
                }
                _ => respond(&mut stream, 404, "application/json", b"{\"error\":\"unknown cmd\"}"),
            }
        }
        _ => respond(&mut stream, 404, "application/json", b"{\"error\":\"not found\"}"),
    }
}

fn respond(stream: &mut TcpStream, code: u16, content_type: &str, body: &[u8]) {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        code, reason, content_type, body.len(),
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}

/// 原子写命令文件：先写临时文件再 rename，避免导航进程读到半截内容。
fn write_cmd(cmd: &str) -> std::io::Result<()> {
    let tmp = format!("{}.tmp", CMD_FILE);
    fs::write(&tmp, cmd)?;
    fs::rename(&tmp, CMD_FILE)?;
    eprintln!("[ctrl-serve] 命令: {}", cmd);
    Ok(())
}
