output "resource_group" {
  value = azurerm_resource_group.rg.name
}

output "storage_account" {
  value = azurerm_storage_account.datalake.name
}

output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}
