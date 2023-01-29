# Order Processing API

## URL

`https://api.example.com/order`

## Phương thức HTTP

`POST`

## Định dạng dữ liệu

### Đầu vào

| Tên | Kiểu | Mô tả |
| ---- | ---- | ------ |
| customer_id | integer | ID của khách hàng đặt hàng |
| product_id | integer | ID của sản phẩm đặt hàng |
| quantity | integer | Số lượng sản phẩm đặt hàng |

### Đầu ra

| Tên | Kiểu | Mô tả |
| ---- | ---- | ------ |
| order_id | integer | ID của đơn đặt hàng |
| status | string | Trạng thái của đơn đặt hàng |

## Mô tả chi tiết

### `/order`

#### Mô tả

Tạo một đơn đặt hàng mới.

#### Tham số

| Tên | Kiểu | Bắt buộc | Mô tả |
| ---- | ---- | -------- | ------ |
| customer_id | integer | Có | ID của khách hàng đặt hàng |
| product_id | integer | Có | ID của sản phẩm đặt hàng |
| quantity | integer | Có | Số lượng sản phẩm đặt hàng |

#### Kết quả trả về

- `200 OK`: Đặt hàng thành công. Body trả về chứa thông tin của đơn đặt hàng.
- `400 Bad Request`: Yêu cầu không hợp lệ hoặc thiếu tham số. Body trả về chứa thông báo lỗi chi tiết.
- `500 Internal Server Error`: Lỗi h
