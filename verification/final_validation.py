#!/usr/bin/env python3
# verification/final_validation.py

import sys
import os

def main():
    """最终验证入口"""
    print("🎯 TURRET Final Validation Suite")
    print("=" * 50)
    
    # 运行接口验证
    from interface_validator import InterfaceValidator
    interface_validator = InterfaceValidator()
    interface_results = interface_validator.validate_all_components()
    
    print("\n" + "=" * 50)
    
    # 运行集成测试
    from integration_test import IntegrationTester
    integration_tester = IntegrationTester()
    integration_results = integration_tester.run_integration_tests()
    
    # 汇总结果
    all_interface_passed = all(interface_results.values())
    all_integration_passed = all(integration_results.values())
    overall_passed = all_interface_passed and all_integration_passed
    
    print("\n" + "=" * 50)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 50)
    print(f"Interface Validation: {sum(interface_results.values())}/{len(interface_results)} passed")
    print(f"Integration Tests: {sum(integration_results.values())}/{len(integration_results)} passed")
    print(f"Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
    
    if overall_passed:
        print("\n🎉 All validations passed! TURRET system is ready for experiments.")
    else:
        print("\n⚠️ Some validations failed. Please check the reports above.")
    
    return 0 if overall_passed else 1

if __name__ == "__main__":
    sys.exit(main())