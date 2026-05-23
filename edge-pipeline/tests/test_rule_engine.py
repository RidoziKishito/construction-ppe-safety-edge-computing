import unittest
import sys
import os

# Trỏ đường dẫn ngược ra ngoài thư mục gốc để import rule_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rule_engine


class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        # Dữ liệu giả lập (Mock data) cho Zone
        self.zones_config = {
            "zones": [
                {
                    "id": "Z_DANGER",
                    "type": "danger_zone",
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }
            ]
        }

    def test_point_in_polygon(self):
        # Điểm (5, 5) nằm bên trong hộp 10x10 -> Phải trả về True
        self.assertTrue(
            rule_engine.point_in_polygon(
                (5, 5), self.zones_config["zones"][0]["polygon"]
            )
        )
        # Điểm (15, 15) nằm ngoài hộp -> Phải trả về False
        self.assertFalse(
            rule_engine.point_in_polygon(
                (15, 15), self.zones_config["zones"][0]["polygon"]
            )
        )

    def test_critical_alert_in_danger_zone(self):
        active_zones = rule_engine.get_person_zones((5, 5), self.zones_config)
        violations = ["no_helmet"]  # Giả lập đang thiếu nón
        alert = rule_engine.classify_alert(active_zones, violations)
        # Đứng trong Danger Zone thì phải báo CRITICAL
        self.assertEqual(alert, "CRITICAL")

    def test_normal_alert_outside_zone(self):
        active_zones = rule_engine.get_person_zones((15, 15), self.zones_config)
        violations = []  # Đầy đủ đồ
        alert = rule_engine.classify_alert(active_zones, violations)
        # Đứng ngoài zone và đủ đồ thì phải NORMAL
        self.assertEqual(alert, "NORMAL")


if __name__ == "__main__":
    unittest.main()
